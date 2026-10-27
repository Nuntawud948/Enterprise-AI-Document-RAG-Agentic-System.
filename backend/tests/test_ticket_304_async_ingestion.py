import asyncio
import time
import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import VectorServiceError
from app.core.security import (
    AuthenticatedUser,
    ClassificationLevel,
    create_access_token,
)
from app.main import app
from app.modules.auth.models.user import UserRole
from app.modules.documents.dtos.requests.document_requests import (
    DocumentCreateRequest,
    DocumentFilterRequest,
    DocumentUploadRequest,
)
from app.modules.documents.interfaces.i_document_repository import IDocumentRepository
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.services.document_service import DocumentService
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import ChunkResult, TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder
from app.shared.tasks.ingestion_tasks import process_document_background_task

client = TestClient(app)


class MockAsyncSessionFactory:
    """Async context manager helper providing an isolated mock AsyncSession."""
    def __init__(self, session: AsyncSession):
        self.session = session

    def __call__(self) -> "MockAsyncSessionFactory":
        return self

    async def __aenter__(self) -> AsyncSession:
        return self.session

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ============================================================================
# 1. Sub-100ms Response Time SLA Verification
# ============================================================================

def test_endpoint_ingestion_sub_100ms_returns_queued():
    """
    Verify POST /api/v1/documents returns HTTP 201 with status QUEUED within 100ms.
    Starlette TestClient executes BackgroundTasks synchronously on the event loop;
    mocking the background task isolates the controller/service latency from the AI pipeline.
    """
    user_id = uuid.uuid4()
    token = create_access_token(
        subject=user_id,
        department="ENGINEERING",
        role=UserRole.MEMBER,
        clearance=ClassificationLevel.INTERNAL,
    )

    # Payload with large text (100KB)
    large_payload = {
        "title": "Sub-100ms SLA Spec",
        "file_name": "large_spec.txt",
        "file_type": "txt",
        "file_size_bytes": 102400,
        "classification_level": "INTERNAL",
        "content": "Enterprise AI architecture specification. " * 2500,
    }

    from app.modules.documents.controllers.documents_controller import get_document_service
    from app.modules.documents.interfaces.i_document_service import IDocumentService

    mock_doc_service = AsyncMock(spec=IDocumentService)
    mock_doc_service.create_document.return_value = MagicMock(
        id=uuid.uuid4(),
        title="Sub-100ms SLA Spec",
        file_name="large_spec.txt",
        file_type="txt",
        file_size_bytes=102400,
        status="QUEUED",
        department="ENGINEERING",
        classification_level="INTERNAL",
        uploader_id=user_id,
        chunks_count=0,
        created_at="2026-10-27T10:00:00Z",
        updated_at="2026-10-27T10:00:00Z",
    )
    app.dependency_overrides[get_document_service] = lambda: mock_doc_service

    try:
        start_time = time.perf_counter()
        response = client.post(
            "/api/v1/documents",
            json=large_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert response.status_code == 201
        body = response.json()
        assert body["code"] == 201
        assert body["status"] == "success"
        assert body["data"]["status"] == "QUEUED"
        assert duration_ms < 100.0, f"Endpoint took {duration_ms:.2f}ms, exceeding 100ms SLA!"
    finally:
        app.dependency_overrides.pop(get_document_service, None)


# ============================================================================
# 2. Full Background Task Execution Flow (QUEUED -> PROCESSING -> SUCCESS)
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_progression_queued_to_processing_to_success():
    """Verify document transitions QUEUED -> PROCESSING -> SUCCESS and commits chunks."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Async Ingestion Test",
        file_name="test_doc.txt",
        file_type="txt",
        file_size_bytes=500,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
        classification_level="INTERNAL",
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    # Chunker & Embedder
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.1] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    raw_text = "This is a comprehensive test of asynchronous document ingestion in background tasks."
    await process_document_background_task(
        document_id=doc_id,
        content=raw_text,
        session_factory=session_factory,
        chunker=chunker,
        embedder=embedder,
    )

    # Verify status progression
    assert doc.status == "SUCCESS"
    assert mock_session.commit.await_count >= 2  # Once for PROCESSING, once for SUCCESS
    mock_session.add_all.assert_called_once()

    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0
    for chunk in saved_chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == doc_id
        assert chunk.department == "ENGINEERING"
        assert chunk.embedding == [0.1] * 768


# ============================================================================
# 3. Markdown Document Ingestion (Header Hierarchy & Offsets)
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_markdown_chunking_and_metadata():
    """Verify Markdown documents use split_markdown and store header metadata."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Markdown Architecture Guide",
        file_name="architecture.md",
        file_type="md",
        file_size_bytes=600,
        status="QUEUED",
        department="RESEARCH",
        uploader_id=uuid.uuid4(),
        classification_level="INTERNAL",
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.05] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    md_content = (
        "# System Architecture\n\n"
        "This section describes high level design.\n\n"
        "## Vector Database\n\n"
        "PostgreSQL pgvector is utilized for cosine similarity queries."
    )

    await process_document_background_task(
        document_id=doc_id,
        content=md_content,
        session_factory=session_factory,
        chunker=chunker,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) >= 2

    # Verify header context is preserved in chunk metadata
    first_chunk = saved_chunks[0]
    assert "headers" in first_chunk.chunk_metadata
    assert first_chunk.chunk_metadata["headers"].get("h1") == "System Architecture"

    second_chunk = saved_chunks[1]
    assert "headers" in second_chunk.chunk_metadata
    assert second_chunk.chunk_metadata["headers"].get("h2") == "Vector Database"

    # Verify character offsets
    for chunk in saved_chunks:
        assert "start_char" in chunk.chunk_metadata
        assert "end_char" in chunk.chunk_metadata
        assert chunk.chunk_metadata["start_char"] < chunk.chunk_metadata["end_char"]
        assert chunk.department == "RESEARCH"


# ============================================================================
# 4. Multi-Tenancy & Department Isolation
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("test_department", ["FINANCE", "LEGAL", "HR", "OPERATIONS"])
async def test_background_task_department_isolation_on_chunks(test_department: str):
    """Verify 100% of persisted chunks inherit parent document's department strictly."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title=f"{test_department} Secret Policy",
        file_name=f"{test_department.lower()}_policy.txt",
        file_type="txt",
        file_size_bytes=300,
        status="QUEUED",
        department=test_department,
        uploader_id=uuid.uuid4(),
        classification_level="CONFIDENTIAL",
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.2] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    content = f"Confidential data belonging strictly to the {test_department} division."
    await process_document_background_task(
        document_id=doc_id,
        content=content,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0

    # Strict multi-tenancy verification
    for chunk in saved_chunks:
        assert chunk.department == test_department
        assert chunk.department == doc.department
        assert chunk.chunk_metadata.get("department") == test_department


# ============================================================================
# 5. Error Handling & Rollback (Transitions to FAILED)
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_embedding_failure_transitions_to_failed():
    """Verify embedder exception triggers rollback, sets status FAILED, and does not leak."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Failing Ingestion",
        file_name="failing.txt",
        file_type="txt",
        file_size_bytes=400,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    # Embedder fails with VectorServiceError
    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.side_effect = VectorServiceError("Gemini API connection timed out")
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    # Should not raise exception
    await process_document_background_task(
        document_id=doc_id,
        content="This document will fail during vector embedding.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"
    assert mock_session.commit.await_count >= 2  # Once for PROCESSING, once for FAILED


@pytest.mark.asyncio
async def test_background_task_db_persistence_failure_transitions_to_failed():
    """Verify DB flush error triggers rollback and marks document as FAILED."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="DB Failure Doc",
        file_name="db_fail.txt",
        file_type="txt",
        file_size_bytes=200,
        status="QUEUED",
        department="FINANCE",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    # Flush fails on bulk chunk persistence
    mock_session.flush.side_effect = SQLAlchemyError("Database disk full or constraint violation")
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.3] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        content="Persisting this content will trigger database flush failure.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"


# ============================================================================
# 6. Empty / Whitespace Content Handling
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t  \n", None])
async def test_background_task_empty_content_graceful_success(empty_content: Optional[str]):
    """Verify empty or whitespace content transitions to SUCCESS without creating chunks."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Empty File",
        file_name="empty.txt",
        file_type="txt",
        file_size_bytes=0,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        content=empty_content,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    mock_session.add_all.assert_not_called()
    mock_gemini.generate_embedding.assert_not_called()


# ============================================================================
# 7. Null Byte Sanitization
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_null_byte_sanitization():
    """Verify null bytes (\x00) are stripped from text to protect PostgreSQL string columns."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Null Byte Sanitization Test",
        file_name="dirty.txt",
        file_type="txt",
        file_size_bytes=100,
        status="QUEUED",
        department="SECURITY",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.4] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    raw_text_with_nulls = "Secure\x00Token\x00Authentication\x00Payload"
    await process_document_background_task(
        document_id=doc_id,
        content=raw_text_with_nulls,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    for chunk in saved_chunks:
        assert "\x00" not in chunk.content
        assert "SecureTokenAuthenticationPayload" in chunk.content


# ============================================================================
# 8. File Path Fallback Ingestion
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_file_path_fallback(tmp_path):
    """Verify background task reads doc.file_path if raw content is None."""
    sample_file = tmp_path / "sample_policy.txt"
    sample_file.write_text("Company policy extracted from uploaded disk file.", encoding="utf-8")

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Disk File Doc",
        file_name="sample_policy.txt",
        file_type="txt",
        file_size_bytes=100,
        status="QUEUED",
        department="HR",
        uploader_id=uuid.uuid4(),
    )
    doc.file_path = str(sample_file)

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.15] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        content=None,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0
    assert "Company policy extracted" in saved_chunks[0].content


# ============================================================================
# 9. Document Commit Latency Polling Retry
# ============================================================================

@pytest.mark.asyncio
async def test_background_task_retry_polling_handles_commit_latency():
    """Verify worker retries fetching doc up to 3 times before declaring missing."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Delayed Commit Doc",
        file_name="delayed.txt",
        file_type="txt",
        file_size_bytes=50,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    # First get returns None (uncommitted), second returns doc
    mock_session.get.side_effect = [None, doc, doc]
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.1] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        content="Successfully fetched on retry attempt.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert mock_session.get.call_count == 2
    assert doc.status == "SUCCESS"


@pytest.mark.asyncio
async def test_background_task_nonexistent_document_graceful_exit():
    """Verify non-existent document ID logs warning and exits without throwing exception."""
    doc_id = uuid.uuid4()
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = None
    session_factory = MockAsyncSessionFactory(mock_session)

    await process_document_background_task(
        document_id=doc_id,
        content="Content for non-existent document.",
        session_factory=session_factory,
    )

    assert mock_session.get.call_count == 3
    mock_session.commit.assert_not_called()
    mock_session.add_all.assert_not_called()


# ============================================================================
# 10. Service Integration & Backward Compatibility
# ============================================================================

@pytest.mark.asyncio
async def test_document_service_enqueues_background_task_and_commits_session():
    """Verify DocumentService.create_document commits session and enqueues background task."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_repo = AsyncMock(spec=IDocumentRepository)
    mock_repo.session = mock_session

    created_doc = Document(
        id=uuid.uuid4(),
        title="Background Task Service Test",
        file_name="test.txt",
        file_type="txt",
        file_size_bytes=1024,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )
    mock_repo.create.return_value = created_doc

    service = DocumentService(doc_repo=mock_repo)
    req = DocumentCreateRequest(
        title="Background Task Service Test",
        file_name="test.txt",
        file_type="txt",
        file_size_bytes=1024,
        content="Direct text payload for ingestion",
    )

    mock_bg_tasks = MagicMock()
    result = await service.create_document(
        request=req,
        department="ENGINEERING",
        uploader_id=created_doc.uploader_id,
        background_tasks=mock_bg_tasks,
    )

    assert result.status == "QUEUED"
    # Verify session commit was explicitly awaited before enqueuing
    mock_session.commit.assert_awaited_once()
    # Verify background task was added
    mock_bg_tasks.add_task.assert_called_once_with(
        process_document_background_task,
        document_id=created_doc.id,
        department="ENGINEERING",
        content="Direct text payload for ingestion",
    )


@pytest.mark.asyncio
async def test_document_service_backward_compatible_without_background_tasks():
    """Verify DocumentService operates without background_tasks parameter (existing test compatibility)."""
    mock_repo = AsyncMock(spec=IDocumentRepository)
    created_doc = Document(
        id=uuid.uuid4(),
        title="Legacy Call",
        file_name="legacy.txt",
        file_type="txt",
        file_size_bytes=100,
        status="QUEUED",
        department="HR",
        uploader_id=uuid.uuid4(),
    )
    mock_repo.create.return_value = created_doc

    service = DocumentService(doc_repo=mock_repo)
    req = DocumentCreateRequest(
        title="Legacy Call",
        file_name="legacy.txt",
        file_type="txt",
        file_size_bytes=100,
    )

    result = await service.create_document(
        request=req,
        department="HR",
        uploader_id=created_doc.uploader_id,
    )

    assert result.status == "QUEUED"
    assert result.title == "Legacy Call"


def test_document_filter_and_upload_request_dtos():
    """Verify DocumentFilterRequest allows SUCCESS status and DocumentUploadRequest alias exists."""
    filter_req = DocumentFilterRequest(status="SUCCESS")
    assert filter_req.status == "SUCCESS"

    upload_req = DocumentUploadRequest(
        title="Alias Test",
        file_name="alias.md",
        file_type="md",
        file_size_bytes=256,
        content="# Hello World",
    )
    assert upload_req.title == "Alias Test"
    assert upload_req.content == "# Hello World"
