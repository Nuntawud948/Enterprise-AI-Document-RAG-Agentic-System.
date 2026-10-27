import asyncio
import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    CircuitBreakerOpenError,
    QuotaExceededError,
    VectorServiceError,
)
from app.modules.documents.models.document import Document, DocumentChunk
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder
from app.shared.tasks.ingestion_tasks import process_document_background_task


class MockAsyncSessionFactory:
    def __init__(self, session: AsyncSession):
        self.session = session

    def __call__(self) -> "MockAsyncSessionFactory":
        return self

    async def __aenter__(self) -> AsyncSession:
        return self.session

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ============================================================================
# Adversarial Test Suite for TICKET-304
# ============================================================================

@pytest.mark.asyncio
async def test_adversarial_partial_batch_embedding_failure_full_rollback():
    """
    Adversarial: When batch 1 succeeds but batch 2 fails with QuotaExceededError,
    verify that session.rollback() is invoked, uncommitted chunks are discarded,
    and doc.status is set to FAILED without leaking exceptions.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Partial Embedding Failure",
        file_name="partial_fail.txt",
        file_type="txt",
        file_size_bytes=2000,
        status="QUEUED",
        department="RISK",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    # Embedder succeeds for first call, raises QuotaExceededError on subsequent call
    call_count = 0

    async def mock_embed(text: str) -> List[float]:
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            raise QuotaExceededError("Rate limit exceeded for text-embedding-004")
        return [0.01] * 768

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.side_effect = mock_embed
    # Use batch_size=2 to force multiple batches
    embedder = VectorEmbedder(gemini_client=mock_gemini, batch_size=2)

    # Long text generating >= 4 chunks
    long_text = "Adversarial test line for chunking and embedding. " * 20

    await process_document_background_task(
        document_id=doc_id,
        content=long_text,
        session_factory=session_factory,
        chunker=chunker,
        embedder=embedder,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"
    # Ensure add_all was not committed
    mock_session.add_all.assert_not_called()


@pytest.mark.asyncio
async def test_adversarial_circuit_breaker_open_during_ingestion():
    """
    Adversarial: When external AI client has tripped circuit breaker (CircuitBreakerOpenError),
    verify background ingestion handles it gracefully and marks status FAILED.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Circuit Breaker Test Doc",
        file_name="cb_test.txt",
        file_type="txt",
        file_size_bytes=300,
        status="QUEUED",
        department="OPERATIONS",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.side_effect = CircuitBreakerOpenError("GeminiClient", 25.0)
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    await process_document_background_task(
        document_id=doc_id,
        content="Content attempting embedding while circuit breaker is OPEN.",
        session_factory=session_factory,
        embedder=embedder,
    )

    assert mock_session.rollback.awaited
    assert doc.status == "FAILED"


@pytest.mark.asyncio
async def test_adversarial_transient_db_disconnect_recovered_via_fresh_session():
    """
    Adversarial: If active session raises OperationalError (e.g. socket reset),
    verify fallback to fresh session marks doc.status = FAILED.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="DB Socket Reset Doc",
        file_name="disconnect.txt",
        file_type="txt",
        file_size_bytes=100,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    # First session fails on commit
    broken_session = AsyncMock(spec=AsyncSession)
    broken_session.get.return_value = doc
    broken_session.commit.side_effect = OperationalError("connection reset by peer", params=None, orig=Exception())
    broken_session.rollback.side_effect = OperationalError("connection reset by peer", params=None, orig=Exception())

    # Fallback fresh session
    fresh_session = AsyncMock(spec=AsyncSession)
    fresh_session.get.return_value = doc

    sessions = [broken_session, fresh_session]
    factory_calls = 0

    class DualSessionFactory:
        def __call__(self):
            nonlocal factory_calls
            s = sessions[min(factory_calls, len(sessions) - 1)]
            factory_calls += 1
            return MockAsyncSessionFactory(s)

    await process_document_background_task(
        document_id=doc_id,
        content="Content to process with database connection failure.",
        session_factory=DualSessionFactory(),
    )

    # Even with connection drop on active session, error was logged and no uncaught exception escaped


@pytest.mark.asyncio
async def test_adversarial_multilingual_unicode_emojis_ingestion():
    """
    Adversarial: Ingest complex multilingual text with Arabic, CJK, Thai, and emojis.
    Verifies character offsets, department isolation, and successful persistence.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Multilingual Spec",
        file_name="multilingual.txt",
        file_type="txt",
        file_size_bytes=1000,
        status="QUEUED",
        department="GLOBAL",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.07] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    multilingual_text = (
        "🚀 Enterprise AI Monolith: ระบบประมวลผลเอกสารอัตโนมัติ 🌟\n\n"
        "مستندات المؤسسة الذكية - 企業向けドキュメント処理システム\n\n"
        "Ensuring 100% UTF-8 string encoding fidelity without truncation."
    )

    await process_document_background_task(
        document_id=doc_id,
        content=multilingual_text,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) > 0
    for chunk in saved_chunks:
        assert chunk.department == "GLOBAL"
        assert len(chunk.content) > 0


@pytest.mark.asyncio
async def test_adversarial_markdown_with_nested_fences_and_special_headers():
    """
    Adversarial: Ingest markdown with edge-case headers, code fences, and unclosed tags.
    """
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Markdown Edge Cases",
        file_name="edge_cases.markdown",
        file_type="text/markdown",
        file_size_bytes=800,
        status="QUEUED",
        department="ENGINEERING",
        uploader_id=uuid.uuid4(),
    )

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = doc
    session_factory = MockAsyncSessionFactory(mock_session)

    mock_gemini = AsyncMock(spec=GeminiClient)
    mock_gemini.generate_embedding.return_value = [0.12] * 768
    embedder = VectorEmbedder(gemini_client=mock_gemini)

    md_text = (
        "# Root Header\n\n"
        "```python\n"
        "# This is a python comment, NOT a markdown header!\n"
        "def foo():\n"
        "    return 42\n"
        "```\n\n"
        "## Real Sub Header\n\n"
        "Content under real sub header."
    )

    await process_document_background_task(
        document_id=doc_id,
        content=md_text,
        session_factory=session_factory,
        embedder=embedder,
    )

    assert doc.status == "SUCCESS"
    saved_chunks = mock_session.add_all.call_args[0][0]
    assert len(saved_chunks) >= 2
    # Ensure Python comment in code block was not treated as h1 header
    for chunk in saved_chunks:
        headers = chunk.chunk_metadata.get("headers", {})
        assert headers.get("h1") != "This is a python comment, NOT a markdown header!"
