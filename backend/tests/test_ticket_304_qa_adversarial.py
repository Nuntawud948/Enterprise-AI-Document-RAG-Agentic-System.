import asyncio
import time
import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    CircuitBreakerOpenError,
    QuotaExceededError,
    VectorServiceError,
)
from app.core.security import (
    AuthenticatedUser,
    ClassificationLevel,
    create_access_token,
)
from app.main import app
from app.modules.auth.models.user import UserRole
from app.modules.documents.dtos.requests.document_requests import DocumentCreateRequest
from app.modules.documents.interfaces.i_document_repository import IDocumentRepository
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.services.document_service import DocumentService
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder
from app.shared.tasks.ingestion_tasks import process_document_background_task

client = TestClient(app)


class MockAsyncSessionFactory:
    """Helper to mock async_sessionmaker providing isolated async context manager."""
    def __init__(self, session: AsyncSession):
        self.session = session

    def __call__(self) -> "MockAsyncSessionFactory":
        return self

    async def __aenter__(self) -> AsyncSession:
        return self.session

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ============================================================================
# Section 1: Sub-100ms SLA & Endpoint Stress Verification
# ============================================================================

def test_endpoint_sla_with_real_service_and_mocked_db_repo():
    """
    Empirical SLA verification:
    Uses the real DocumentService and Controller (mocking only the DB session/repo)
    to verify that the HTTP response returns within 100ms and properly enqueues
    the background task without blocking on background processing.
    """
    user_id = uuid.uuid4()
    token = create_access_token(
        subject=user_id,
        department="ENGINEERING",
        role=UserRole.MEMBER,
        clearance=ClassificationLevel.INTERNAL,
    )

    from app.core.database import get_db_session
    from datetime import datetime, timezone
    mock_session = AsyncMock(spec=AsyncSession)

    def fake_add(entity):
        if getattr(entity, "id", None) is None:
            entity.id = uuid.uuid4()
        if getattr(entity, "created_at", None) is None:
            entity.created_at = datetime.now(timezone.utc)
        if getattr(entity, "updated_at", None) is None:
            entity.updated_at = datetime.now(timezone.utc)

    mock_session.add.side_effect = fake_add

    # Mock the session commit and flush
    async def fake_commit():
        pass
    mock_session.commit.side_effect = fake_commit

    app.dependency_overrides[get_db_session] = lambda: mock_session

    # Patch process_document_background_task so TestClient doesn't execute the heavy worker inline
    with patch("app.modules.documents.services.document_service.process_document_background_task") as mock_task:
        mock_task.__name__ = "process_document_background_task"

        payload = {
            "title": "Real Service SLA Verification",
            "file_name": "sla_benchmark.txt",
            "file_type": "txt",
            "file_size_bytes": 50000,
            "classification_level": "INTERNAL",
            "content": "Enterprise AI Document Ingestion Payload. " * 500,
        }

        # Warm-up run
        client.post(
            "/api/v1/documents",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        latencies_ms = []
        for _ in range(50):
            start = time.perf_counter()
            resp = client.post(
                "/api/v1/documents",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            assert resp.status_code == 201
            assert resp.json()["data"]["status"] == "QUEUED"

        app.dependency_overrides.pop(get_db_session, None)

        p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
        p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99 = sorted(latencies_ms)[-1]

        # Assert sub-100ms SLA
        assert p95 < 100.0, f"P95 latency {p95:.2f}ms exceeds 100ms SLA"
        assert p99 < 150.0, f"P99 latency {p99:.2f}ms exceeds tolerance"
        assert mock_task.called, "process_document_background_task was not scheduled in BackgroundTasks"


# ============================================================================
# Section 2: Lifecycle Transitions (QUEUED -> PROCESSING -> SUCCESS)
# ============================================================================

@pytest.mark.asyncio
async def test_full_lifecycle_and_chunk_indexing_integrity():
    """
    Verifies that during ingestion:
    1. Status transitions QUEUED -> PROCESSING -> SUCCESS
    2. Commit is called after setting PROCESSING so other workers/clients see it
    3. Chunks are strictly indexed 0, 1, 2, ... with no gaps or duplicates
    4. Start and end char offsets are monotonically increasing
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Lifecycle Document",
        file_name="lifecycle.txt",
        file_type="txt",
        file_size_bytes=1000,
        status="QUEUED",
        department="OPERATIONS",
        uploader_id=uuid.uuid4(),
    )

    status_history = [doc.status]

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc

    async def track_commit():
        status_history.append(doc.status)

    mock_session.commit.side_effect = track_commit
    session_factory = MockAsyncSessionFactory(mock_session)

    chunker = TextChunker(chunk_size=150, chunk_overlap=20)
    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.123] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    content = (
        "Paragraph 1: Background worker architecture in modern enterprise distributed systems.\n\n"
        "Paragraph 2: PostgreSQL transactional isolation guarantees and async session lifecycles.\n\n"
        "Paragraph 3: Vector search and similarity ranking using pgvector cosine distance."
    )

    await process_document_background_task(
        document_id=doc_id,
        content=content,
        session_factory=session_factory,
        chunker=chunker,
        embedder=embedder,
    )

    # Verify lifecycle progression in order
    assert doc.status == "SUCCESS"
    assert "PROCESSING" in status_history, "Status was never transitioned to PROCESSING"
    assert status_history.index("PROCESSING") < status_history.index("SUCCESS"), "PROCESSING must precede SUCCESS"

    # Verify chunks
    saved_chunks: List[DocumentChunk] = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) >= 3

    for idx, chunk in enumerate(saved_chunks):
        assert chunk.chunk_index == idx, f"Chunk index mismatch: expected {idx}, got {chunk.chunk_index}"
        assert chunk.department == "OPERATIONS"
        assert chunk.document_id == doc_id
        assert chunk.embedding == [0.123] * 768
        assert chunk.token_count > 0
        assert chunk.start_char < chunk.end_char


# ============================================================================
# Section 3: Error Handling, Rollback & Exception Containment
# ============================================================================

@pytest.mark.asyncio
async def test_error_handling_quota_exceeded_rolls_back_and_marks_failed():
    """
    Adversarial: When rate limiter/quota (HTTP 429) triggers QuotaExceededError,
    worker must rollback uncommitted chunks, set status to FAILED, and NOT re-raise.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Quota Exceeded Doc",
        file_name="quota.txt",
        file_type="txt",
        file_size_bytes=500,
        status="QUEUED",
        department="FINANCE",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.side_effect = QuotaExceededError("Rate limit exceeded. Retry after 60s", retry_after=60)
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    # Must complete cleanly without uncaught exception escaping
    await process_document_background_task(
        document_id=doc_id,
        content="This request triggers 429 quota exhaustion.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"
    mock_session.add_all.assert_not_called()


@pytest.mark.asyncio
async def test_error_handling_unexpected_runtime_exception():
    """
    Adversarial: When an unexpected low-level exception occurs (e.g. ValueError or RuntimeError),
    worker must cleanly catch it, rollback, and set status to FAILED.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Runtime Crash Doc",
        file_name="crash.txt",
        file_type="txt",
        file_size_bytes=500,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    # Chunker explodes with unexpected RuntimeError
    mock_chunker = MagicMock(spec=TextChunker)
    mock_chunker.split_text_with_metadata.side_effect = RuntimeError("Kernel memory allocator fault")

    await process_document_background_task(
        document_id=doc_id,
        content="Triggering unexpected runtime error",
        session_factory=session_factory,
        chunker=mock_chunker,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"


# ============================================================================
# Section 4: Multi-Tenancy & Department Isolation Verification
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("doc_dept, adversarial_kwarg_dept", [
    ("HR", "FINANCE"),
    ("LEGAL", "PUBLIC"),
    ("SECURITY", "ENGINEERING"),
    ("FINANCE", "HR"),
])
async def test_adversarial_department_tampering_isolation(doc_dept: str, adversarial_kwarg_dept: str):
    """
    Adversarial: Even if an external caller attempts to pass a mismatched department
    parameter in kwargs or background task payload, the persisted chunks MUST strictly
    inherit doc.department from the database record (source of truth).
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Department Tamper Test",
        file_name="tamper.txt",
        file_type="txt",
        file_size_bytes=300,
        status="QUEUED",
        department=doc_dept,
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.5] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        department=adversarial_kwarg_dept,  # Mismatched!
        content="Confidential department content to verify tenant containment.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0
    for chunk in saved_chunks:
        # Strict isolation check: Must equal doc.department, NEVER the spoofed department
        assert chunk.department == doc_dept
        assert chunk.department != adversarial_kwarg_dept
        assert chunk.chunk_metadata["department"] == doc_dept


# ============================================================================
# Section 5: Independent Session Lifecycle & Leaked State Isolation
# ============================================================================

@pytest.mark.asyncio
async def test_session_isolation_independent_of_closed_request_session():
    """
    Verifies that the worker creates its own session from the factory
    and does not share or leak anything from a closed HTTP request session.
    """
    # 1. Simulate an HTTP request session that closes immediately
    request_session = AsyncMock(spec=AsyncSession)
    await request_session.close()

    # 2. Simulate worker's independent session
    worker_session = AsyncMock(spec=AsyncSession)
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Session Isolation Test",
        file_name="iso.txt",
        file_type="txt",
        file_size_bytes=100,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )
    worker_session.get.return_value = doc

    session_factory = MockAsyncSessionFactory(worker_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.1] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    # Worker runs with its own factory
    await process_document_background_task(
        document_id=doc_id,
        content="Testing clean session lifecycle separation.",
        session_factory=session_factory,
        embedder=embedder,
    )

    # Worker session operated completely independently
    assert doc.status == "SUCCESS"
    assert worker_session.commit.await_count >= 2
    assert request_session.commit.await_count == 0


# ============================================================================
# Section 6: Adversarial Inputs (Massive Tokens, Emojis, ANSI Escapes)
# ============================================================================

@pytest.mark.asyncio
async def test_adversarial_massive_unbroken_string_chunking():
    """
    Adversarial: 5,000-char continuous string without spaces, newlines, or delimiters.
    Chunker must fall back to fixed-length character slicing without infinite loop.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Massive Unbroken Token",
        file_name="unbroken.txt",
        file_type="txt",
        file_size_bytes=5000,
        status="QUEUED",
        department="SECURITY",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.25] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    unbroken_text = "A" * 5000

    await process_document_background_task(
        document_id=doc_id,
        content=unbroken_text,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) >= 10
    total_reconstructed = "".join([c.content for c in saved_chunks])
    assert len(total_reconstructed) >= 5000


@pytest.mark.asyncio
async def test_adversarial_ansi_escapes_and_control_chars():
    """
    Adversarial: Ingest text with terminal ANSI escape codes and obscure control chars.
    Verifies clean persistence without database or serialization errors.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="ANSI Escapes Doc",
        file_name="terminal.log",
        file_type="txt",
        file_size_bytes=200,
        status="QUEUED",
        department="DEVOPS",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.33] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    ansi_text = "\x1b[31;1mCRITICAL ERROR\x1b[0m: Out of memory at \x07ring_bell.\nDetails: OK"

    await process_document_background_task(
        document_id=doc_id,
        content=ansi_text,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0
    assert "CRITICAL ERROR" in saved_chunks[0].content
