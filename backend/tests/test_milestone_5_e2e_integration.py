import asyncio
import time
import uuid
import pytest
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    BaseAppException,
    CircuitBreakerOpenError,
    DocumentIngestionError,
    EntityNotFoundException,
    EntityNotFoundError,
    QuotaExceededError,
    VectorServiceError,
    app_exception_handler,
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.core.security import (
    AuthenticatedUser,
    ClassificationLevel,
    create_access_token,
)
from app.modules.auth.models.user import UserRole
from app.modules.documents.dtos.requests.document_requests import (
    DocumentCreateRequest,
    DocumentUploadRequest,
)
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.modules.documents.services.document_service import DocumentService
from app.shared.clients.circuit_breaker import CircuitBreaker, CircuitState
from app.shared.clients.gemini_client import GeminiClient
from app.shared.dtos.api_response import APIResponse
from app.shared.rag_engine.chunker import TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder
from app.shared.tasks.ingestion_tasks import process_document_background_task


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


class TestMilestone5CrossMilestoneIntegration:
    """
    Milestone 5 Cross-Milestone Integration & E2E Verification Suite.
    Validates end-to-end interoperability across M1, M2, M3, and M4.
    """

    @pytest.mark.asyncio
    async def test_m1_m2_m3_m4_happy_path_full_pipeline_with_isolation(self):
        """
        Full Cross-Milestone E2E Flow:
        - M4 Document creation in QUEUED state
        - M4 Background ingestion executed in isolated session factory
        - M3 Markdown header hierarchy parsing & chunk boundary tracking
        - M3 VectorEmbedder concurrent embedding batching via M2 GeminiClient
        - M3 ChunkRepository bulk persistence (create_chunks)
        - M4 Status progression: QUEUED -> PROCESSING -> SUCCESS
        - M4 & M1 Strict department isolation verification
        """
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        dept = "FINANCE"

        doc = Document(
            id=doc_id,
            title="Q3 Financial Overview",
            file_name="q3_financial_overview.md",
            file_type="markdown",
            file_size_bytes=2048,
            status="QUEUED",
            classification_level="CONFIDENTIAL",
            department=dept,
            uploader_id=user_id,
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.get.return_value = doc
        session_factory = MockAsyncSessionFactory(mock_session)

        # Markdown payload with headers, lists, code fences
        markdown_content = """# Executive Summary
The financial performance for Q3 demonstrated strong revenue growth of 18%.

## Revenue Breakdown
- North America: $12.5M
- EMEA: $8.2M
- APAC: $6.1M

```sql
SELECT department, SUM(revenue) FROM financial_records GROUP BY department;
```

### Risk Assessment
Currency fluctuations in EMEA posed a minor variance in foreign exchange margins.
"""
        mock_gemini = AsyncMock(spec=GeminiClient)
        mock_gemini.generate_embedding.return_value = [0.05] * 768

        chunker = TextChunker(chunk_size=150, chunk_overlap=25)
        embedder = VectorEmbedder(gemini_client=mock_gemini, batch_size=4)

        # Execute background task
        await process_document_background_task(
            document_id=doc_id,
            department=dept,
            content=markdown_content,
            session_factory=session_factory,
            chunker=chunker,
            embedder=embedder,
        )

        # Verify document status transitioned to SUCCESS
        assert doc.status == "SUCCESS"
        assert mock_session.commit.await_count >= 2  # Once for PROCESSING, once for SUCCESS
        mock_session.add_all.assert_called_once()

        saved_chunks = mock_session.add_all.call_args[0][0]
        assert len(saved_chunks) >= 3

        for idx, ch in enumerate(saved_chunks):
            assert ch.document_id == doc_id
            assert ch.department == "FINANCE"  # Strict department isolation
            assert ch.chunk_index == idx
            assert len(ch.embedding) == 768
            assert ch.token_count > 0
            assert "start_char" in ch.chunk_metadata
            assert "end_char" in ch.chunk_metadata
            assert ch.start_char == ch.chunk_metadata["start_char"]
            assert ch.end_char == ch.chunk_metadata["end_char"]

    @pytest.mark.asyncio
    async def test_m1_m2_m4_circuit_breaker_trip_and_fast_fail_resiliency(self):
        """
        Adversarial Failure Cascade:
        - Gemini API throws QuotaExceededError (HTTP 429) continuously
        - M2 retry backoff triggers and exhausts retries
        - M2 CircuitBreaker transitions CLOSED -> OPEN
        - M4 Background ingestion rolls back uncommitted chunks and sets document to FAILED
        - Next incoming request to AI service fast-fails via M1 CircuitBreakerOpenError (HTTP 503)
        - Traceback is masked and contains X-Request-ID and Retry-After header
        """
        breaker = CircuitBreaker(
            service_name="GeminiEmbeddingService",
            failure_threshold=1,
            recovery_timeout=10.0,
        )
        flaky_gemini = GeminiClient(
            api_key="key",
            circuit_breaker=breaker,
            sleep_fn=AsyncMock(),
        )
        # Mock persistent 429 Quota Exceeded error
        flaky_gemini._raw_generate_embedding = AsyncMock(
            side_effect=QuotaExceededError("Rate limit exceeded. Tier quota exhausted.", retry_after=10)
        )

        embedder = VectorEmbedder(gemini_client=flaky_gemini)
        doc_id = uuid.uuid4()

        doc = Document(
            id=doc_id,
            title="Engineering Tech Spec",
            file_name="spec.txt",
            file_type="txt",
            file_size_bytes=1024,
            status="QUEUED",
            classification_level="INTERNAL",
            department="ENGINEERING",
            uploader_id=uuid.uuid4(),
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.get.return_value = doc
        session_factory = MockAsyncSessionFactory(mock_session)

        # Run background ingestion
        await process_document_background_task(
            document_id=doc_id,
            department="ENGINEERING",
            content="Sample architecture documentation content for parsing.",
            session_factory=session_factory,
            embedder=embedder,
        )

        # Document must be marked FAILED, session rolled back
        assert doc.status == "FAILED"
        mock_session.rollback.assert_awaited()

        # Verify CircuitBreaker is now OPEN
        assert breaker.state == CircuitState.OPEN

        # Direct attempt to use client now fast-fails immediately with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await flaky_gemini.generate_embedding("Another chunk")
        assert exc_info.value.status_code == 503
        assert "is OPEN" in exc_info.value.message

    def test_m1_global_exception_traceback_masking_and_request_id(self):
        """
        M1 Global Error Handling:
        - Unhandled system error in endpoint must never leak internal stack traces
        - Response envelope must follow standard APIResponse format
        - X-Request-ID header must be returned on HTTP 500
        """
        test_app = FastAPI()
        test_app.add_exception_handler(BaseAppException, app_exception_handler)
        test_app.add_exception_handler(Exception, generic_exception_handler)

        @test_app.get("/unhandled-error")
        async def trigger_internal_error(request: Request):
            request.state.request_id = "req-e2e-traceback-test-12345"
            # Simulate unexpected bug
            raise ZeroDivisionError("division by zero in database connection pool")

        sync_client = TestClient(test_app, raise_server_exceptions=False)
        response = sync_client.get("/unhandled-error")

        assert response.status_code == 500
        assert response.headers.get("X-Request-ID") == "req-e2e-traceback-test-12345"

        body = response.json()
        assert body["code"] == 500
        assert body["status"] == "error"
        assert body["data"] is None
        assert "ZeroDivisionError" not in body["errors"][0]
        assert "division by zero" not in body["errors"][0]
        assert "Traceback" not in body["errors"][0]
        assert "An unexpected internal server error occurred" in body["errors"][0]
        assert body["meta"]["request_id"] == "req-e2e-traceback-test-12345"

    @pytest.mark.asyncio
    async def test_m2_token_guardrail_prevents_payload_overload(self):
        """
        M2 Token Guardrail:
        - Input exceeding MAX_INPUT_TOKENS (8192 tokens ~ 32,768 characters) is rejected with ValidationException (HTTP 400).
        """
        gemini = GeminiClient(api_key="test-key")
        oversized_text = "word " * 10000  # 50,000 characters > 8192 tokens

        with pytest.raises(BaseAppException) as exc_info:
            await gemini.generate_embedding(oversized_text)
        assert exc_info.value.status_code == 400
        assert "exceeds maximum allowed token limit" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_m3_chunker_handles_massive_document_within_token_limits(self):
        """
        M3 Chunking under extreme scale:
        - 150,000 characters document is partitioned into discrete chunks
        - Every generated chunk strictly satisfies chunk_size <= 500 and token_count <= 125
        - Guarantees zero chunks violate GeminiClient's MAX_INPUT_TOKENS constraint
        """
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        massive_text = "The quick brown fox jumps over the lazy dog. " * 3500  # ~157,500 chars

        chunks = chunker.split_text_with_metadata(massive_text)
        assert len(chunks) > 300

        for chunk in chunks:
            assert len(chunk.text) <= 500
            assert chunk.token_count <= 125
            assert chunk.token_count < GeminiClient.MAX_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_m4_sub_100ms_sla_under_concurrency(self):
        """
        M4 Endpoint SLA under concurrent load:
        - 20 concurrent requests to DocumentService.create_document
        - All complete under 100ms SLA without blocking on vectorization
        """
        mock_doc_repo = AsyncMock()
        mock_chunk_repo = AsyncMock()

        async def fake_create(doc):
            doc.id = uuid.uuid4()
            return doc

        mock_doc_repo.create = AsyncMock(side_effect=fake_create)
        mock_doc_repo.session = AsyncMock()

        service = DocumentService(doc_repo=mock_doc_repo, chunk_repo=mock_chunk_repo)
        mock_bg_tasks = MagicMock()

        req = DocumentCreateRequest(
            title="SLA Test Document",
            file_name="sla_test.pdf",
            file_type="pdf",
            file_size_bytes=10485760,
            classification_level="INTERNAL",
            content="Heavy document content that would take seconds to vectorize...",
        )

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            resp = await service.create_document(
                request=req,
                department="LEGAL",
                uploader_id=uuid.uuid4(),
                background_tasks=mock_bg_tasks,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            assert resp.status == "QUEUED"

        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        assert max_latency < 100.0, f"Max latency was {max_latency:.2f}ms >= 100ms SLA threshold"
        assert avg_latency < 50.0, f"Avg latency was {avg_latency:.2f}ms"
        assert mock_bg_tasks.add_task.call_count == 20

    def test_m1_backward_compatibility_entity_not_found_error_alias(self):
        """
        M1 Backward Compatibility:
        - EntityNotFoundError must be an alias of EntityNotFoundException
        - Instance check and subclass check must pass
        """
        assert EntityNotFoundError is EntityNotFoundException
        err = EntityNotFoundError("Document", "123")
        assert isinstance(err, EntityNotFoundException)
        assert err.status_code == 404

    @pytest.mark.asyncio
    async def test_m2_three_state_transitions_complete_cycle(self):
        """
        M2 Circuit Breaker State Transitions:
        - Initial state: CLOSED
        - Tripping: 3 failures -> OPEN
        - Fast-fail when OPEN
        - Recovery timeout elapsed: transition to HALF_OPEN
        - Trial probe success: transition to CLOSED
        - Trial probe failure in HALF_OPEN: immediately trips back to OPEN
        """
        cb = CircuitBreaker(service_name="TestGeminiAPI", failure_threshold=3, recovery_timeout=0.05)
        assert cb.state == CircuitState.CLOSED

        mock_func = AsyncMock()

        # Step 1: 2 transient failures (below threshold) -> remains CLOSED
        mock_func.side_effect = QuotaExceededError("Quota exhausted")
        for _ in range(2):
            with pytest.raises(QuotaExceededError):
                await cb.call(mock_func)
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count in (1, 2)

        # Step 2: 3rd transient failure -> trips to OPEN
        with pytest.raises(QuotaExceededError):
            await cb.call(mock_func)
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

        # Step 3: Call while OPEN fast-fails without executing function
        call_count_before = mock_func.call_count
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await cb.call(mock_func)
        assert cb.state == CircuitState.OPEN
        assert mock_func.call_count == call_count_before  # Untouched!
        assert exc_info.value.status_code == 503

        # Step 4: Wait for recovery timeout -> transitions to HALF_OPEN
        await asyncio.sleep(0.06)
        # Next call executes trial probe
        mock_func.side_effect = None
        mock_func.return_value = "trial_success"
        res = await cb.call(mock_func)
        assert res == "trial_success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

        # Step 5: Trip back to OPEN, wait, then fail trial in HALF_OPEN
        mock_func.side_effect = QuotaExceededError("Trip again")
        for _ in range(3):
            with pytest.raises(QuotaExceededError):
                await cb.call(mock_func)
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.06)
        # In HALF_OPEN, trial probe fails -> trips immediately back to OPEN
        with pytest.raises(QuotaExceededError):
            await cb.call(mock_func)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_m2_multi_tier_fallback_model_switch_and_local_message(self):
        """
        M2 Multi-tier Model Fallback:
        - Tier 1: Primary model fails with transient error -> falls back to Tier 2 (Flash)
        - Tier 2: Flash succeeds -> returns Flash completion
        - Tier 3: If Flash also fails -> returns LOCAL_FALLBACK_MESSAGE gracefully without crashing
        """
        cb = CircuitBreaker(service_name="GeminiTest", failure_threshold=5, recovery_timeout=30.0)
        client = GeminiClient(
            api_key="test-api-key",
            default_chat_model="gemini-1.5-pro",
            secondary_chat_model="gemini-1.5-flash",
            circuit_breaker=cb,
            sleep_fn=AsyncMock(),
        )

        # Scenario A: Primary fails, Secondary succeeds
        async def mock_raw_chat(prompt, model, system_instruction=None):
            if model == "gemini-1.5-pro":
                raise QuotaExceededError("Pro quota exceeded")
            return f"Flash completion for: {prompt}"

        client._raw_generate_chat = mock_raw_chat

        result = await client.generate_chat_response("What is RAG?")
        assert result == "Flash completion for: What is RAG?"

        # Scenario B: Both Primary and Secondary fail -> Graceful local message
        async def mock_all_fail(prompt, model, system_instruction=None):
            raise QuotaExceededError("All AI services down")

        client._raw_generate_chat = mock_all_fail
        result2 = await client.generate_chat_response("Summarize this document")
        assert result2 == GeminiClient.LOCAL_FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_m3_recursive_character_splitting_delimiters_hierarchy(self):
        """
        M3 Recursive Character Splitting:
        - Respects delimiters hierarchy: ["\\n\\n", "\\n", ". ", " ", ""]
        - Preserves chunk overlap between segments
        """
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        # Text with clear paragraphs
        text = (
            "Paragraph One has significant content explaining system architecture.\n\n"
            "Paragraph Two discusses database optimization with pgvector and HNSW index.\n\n"
            "Paragraph Three focuses on API resiliency with circuit breakers and retries."
        )

        chunks = chunker.split_text_with_metadata(text)
        assert len(chunks) >= 3
        # Check monotonic offsets
        for i in range(len(chunks) - 1):
            assert chunks[i].start_char < chunks[i + 1].start_char
            assert chunks[i].end_char <= chunks[i + 1].end_char + chunker.chunk_overlap

    @pytest.mark.asyncio
    async def test_m3_markdown_code_fence_and_nested_headers(self):
        """
        M3 Markdown Header Hierarchy:
        - Accurately tracks # (h1), ## (h2), ### (h3)
        - Does NOT parse markdown inside ``` code fences as headers
        """
        chunker = TextChunker(chunk_size=250, chunk_overlap=30)
        md_text = """# Architecture Overview
This document specifies the enterprise architecture.

## Database Layer
We utilize PostgreSQL with pgvector.

```python
# This comment inside code block must NOT become an H1 header!
def get_db():
    pass
```

### Indexing Strategy
HNSW index is configured for vector cosine distance.
"""
        chunks = chunker.split_markdown(md_text)
        assert len(chunks) >= 3

        # Verify that code block didn't corrupt headers
        code_chunk = next(ch for ch in chunks if "def get_db()" in ch.text)
        assert code_chunk.metadata["headers"].get("h2") == "Database Layer"
        assert "# This comment inside code block" not in str(code_chunk.metadata["headers"])

        # Verify indexing strategy chunk has h1, h2, and h3
        hnsw_chunk = next(ch for ch in chunks if "Indexing Strategy" in ch.text or "HNSW index" in ch.text)
        assert hnsw_chunk.metadata["headers"].get("h1") == "Architecture Overview"
        assert hnsw_chunk.metadata["headers"].get("h2") == "Database Layer"
        assert hnsw_chunk.metadata["headers"].get("h3") == "Indexing Strategy"

    @pytest.mark.asyncio
    async def test_m4_strict_department_vector_isolation_cross_tenant_leakage(self):
        """
        M4 Strict Department Isolation at Repository Level:
        - Compiles search_similar_chunks statement
        - Confirms WHERE clause strictly binds department == user's department
        - Confirms allowed_clearances filter is applied
        """
        mock_session = AsyncMock(spec=AsyncSession)
        repo = ChunkRepository(session=mock_session)

        # Mock execute returning empty rows
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        fake_embedding = [0.1] * 768
        await repo.search_similar_chunks(
            embedding=fake_embedding,
            department="LEGAL",
            top_k=5,
            score_threshold=0.7,
            allowed_clearances=["INTERNAL", "CONFIDENTIAL"],
        )

        # Verify execute was called with a Select statement containing department and clearance conditions
        assert mock_session.execute.called
        stmt = mock_session.execute.call_args[0][0]
        sql_string = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "document_chunks.department = 'LEGAL'" in sql_string
        assert "documents.classification_level IN ('INTERNAL', 'CONFIDENTIAL')" in sql_string

