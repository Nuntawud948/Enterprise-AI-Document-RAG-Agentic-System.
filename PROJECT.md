# Project: Enterprise AI Document RAG & System Resiliency (Epic 2)

## Architecture
- **Pattern:** Enterprise Modular Monolith with Clean Architecture and Department-based Multi-tenancy.
- **Layers:**
  - `core/`: Global settings, async database engine/sessions, security/JWT/RBAC, and custom exception hierarchy.
  - `shared/`: Cross-cutting concerns including AI Clients (`GeminiClient`, `CircuitBreaker`), RAG Engine (`TextChunker`, `VectorEmbedder`), Background Workers (`ingestion_tasks`), and Shared DTOs (`APIResponse[T]`, `PaginationParams`).
  - `modules/documents/`: Domain slice for Document and DocumentChunk entities, repositories (`IChunkRepository`, `ChunkRepository`), services (`DocumentService`), and API controllers (`documents_controller`).
  - `modules/auth/`, `modules/hr/`: Sibling domain slices (retained with 0 regressions).
  - `tests/`: Pytest suite with hermetic mocking (`AsyncMock`, SQLAlchemy statement compilation).

## Feature Inventory
Every feature identified during the Survey phase is mapped to a specific milestone:
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Custom Domain Exception Hierarchy | Base domain exceptions extending `BaseAppException` (`VectorServiceError`, `QuotaExceededError`, `CircuitBreakerOpenError`, `DocumentIngestionError`) | M1 (T-103) | `app/core/exceptions.py`, survey |
| 2 | EntityNotFoundError Alias | Backward-compatible alias for `EntityNotFoundException` | M1 (T-103) | `mid_level_transformation_sprint.md`, survey |
| 3 | Global Traceback Sanitization | Mask internal tracebacks from HTTP clients, log structured traces with `request_id` | M1 (T-103) | `ORIGINAL_REQUEST.md`, survey |
| 4 | Standardized Error Envelope | Format all domain exceptions and unhandled errors into standard `APIResponse[T]` | M1 (T-103) | `app/shared/dtos/api_response.py`, survey |
| 5 | 3-State Circuit Breaker | State machine (`CLOSED`, `OPEN`, `HALF_OPEN`) tracking failures and fast-failing open circuits | M2 (T-302) | `mid_level_transformation_sprint.md`, survey |
| 6 | Exponential Backoff with Jitter | Retry mechanism with full jitter for transient network/API failures (HTTP 429/503) | M2 (T-302) | `ORIGINAL_REQUEST.md`, survey |
| 7 | Multi-tier Model Fallback | Fallback from Primary model to Secondary model (Flash) to graceful local fallback message | M2 (T-302) | `mid_level_transformation_sprint.md`, survey |
| 8 | Token Usage Guardrail | Enforce `MAX_INPUT_TOKENS` constraint (8,192 tokens) before dispatching remote calls | M2 (T-302) | `mid_level_transformation_sprint.md`, survey |
| 9 | Resilient GeminiClient | Wire Circuit Breaker and retry wrapper into `GeminiClient` for embeddings and generation | M2 (T-302) | `app/shared/clients/gemini_client.py`, survey |
| 10 | Recursive Character Chunking | Hierarchical splitting with delimiters `["\n\n", "\n", ". ", " ", ""]` and overlap validation | M3 (T-301) | `app/shared/rag_engine/chunker.py`, survey |
| 11 | Markdown Header Splitting | Split markdown documents by header hierarchy (`#`, `##`, `###`) preserving header context | M3 (T-301) | `mid_level_transformation_sprint.md`, survey |
| 12 | Chunk Boundary & Token Metadata | Extract character start/end offsets and token counts in structured `ChunkResult` | M3 (T-301) | `app/modules/documents/models/document.py`, survey |
| 13 | Vector Embedding Integration | Generate 768-dimensional normalized embeddings via resilient `GeminiClient` | M3 (T-301) | `app/shared/rag_engine/embedder.py`, survey |
| 14 | Bulk Chunk Persistence | Add `create_chunks` to `IChunkRepository` & `ChunkRepository` for single-transaction inserts | M3 (T-301) | `app/modules/documents/interfaces/i_chunk_repository.py`, survey |
| 15 | Non-blocking Async Ingestion | Offload parsing/chunking/embedding from HTTP request cycle to FastAPI `BackgroundTasks` | M4 (T-304) | `ORIGINAL_REQUEST.md`, survey |
| 16 | Sub-100ms Ingestion Response | Ingestion endpoint returns HTTP 202/201 within 100ms without blocking on vectorization | M4 (T-304) | `ORIGINAL_REQUEST.md`, survey |
| 17 | Ingestion Status Lifecycle | Track document state: `QUEUED` -> `PROCESSING` -> `SUCCESS` / `FAILED` in database | M4 (T-304) | `ORIGINAL_REQUEST.md`, survey |
| 18 | Independent Session Management | Worker runs in isolated `AsyncSessionLocal()`, preventing session conflict after response | M4 (T-304) | `app/core/database.py`, survey |
| 19 | Department Isolation in Chunks | Persist chunks strictly with parent document's department to prevent cross-tenant vector leakage | M4 (T-304) | `app/shared/models/base_entity.py`, survey |

## Milestones
| # | Name | Ticket | Scope | Dependencies | Status |
|---|------|--------|-------|-------------|--------|
| M1 | Global Error Handling & Resiliency | TICKET-103 | `backend/app/core/exceptions.py`, custom exceptions, `main.py` handlers | None | DONE ✅ |
| M2 | AI API Fallback, Circuit Breaker & Rate Limiting | TICKET-302 | `backend/app/shared/clients/circuit_breaker.py`, `gemini_client.py` | M1 | DONE ✅ |
| M3 | RAG Chunking Strategy & Embedding Engine | TICKET-301 | `backend/app/shared/rag_engine/chunker.py`, `embedder.py`, `i_chunk_repository.py`, `chunk_repository.py` | M1, M2 | DONE ✅ |
| M4 | Async Ingestion & Background Task Processing | TICKET-304 | `backend/app/shared/tasks/ingestion_tasks.py`, `backend/app/modules/documents/controllers/documents_controller.py`, `document_service.py` | M1, M2, M3 | DONE ✅ |
| M5 | Final E2E Integration, Adversarial Hardening & Sign-off | All | Verify 100% test pass (baseline + new suites), adversarial edge case coverage, forensic audit | M1, M2, M3, M4 | DONE ✅ |

## Mandatory Governance Pipeline (7-Step Sequence per Ticket)
For every milestone ticket (T-103, T-302, T-301, T-304):
1. **Step 1 (Plan Drafting):** Worker drafts concrete technical design.
2. **Step 2 (Ponytail Pre-Code Review):** Review plan against over-engineering, junior readability ("ปรับ / เพิ่ม / ลด / พอดี").
3. **Step 3 (Plan Alignment):** Worker aligns plan with Ponytail feedback.
4. **Step 4 (Implementation):** Worker implements clean, modular code.
5. **Step 5 (Ponytail Post-Code Review):** Review code diff for bloat, unnecessary complexity, clean interfaces.
6. **Step 6 (QA Adversarial Gate):** QA Tester runs pytest, verifies error handling, rate limits, latency targets.
7. **Step 7 (Tech Lead Sign-off):** Gate check passes, forensic audit verifies integrity, mark milestone DONE.

## Interface Contracts

### M1 (T-103) Core Exceptions Contract
```python
# backend/app/core/exceptions.py
class BaseAppException(Exception):
    message: str
    status_code: int
    errors: List[str]

class EntityNotFoundException(BaseAppException): ...
EntityNotFoundError = EntityNotFoundException

class VectorServiceError(BaseAppException):
    # status_code = 502 Bad Gateway
    def __init__(self, message: str = "Vector embedding service failed.", errors: Optional[List[str]] = None): ...

class QuotaExceededError(BaseAppException):
    # status_code = 429 Too Many Requests
    def __init__(self, message: str = "AI service quota or rate limit exceeded.", retry_after: Optional[int] = None): ...

class CircuitBreakerOpenError(BaseAppException):
    # status_code = 503 Service Unavailable
    def __init__(self, service_name: str, recovery_time_remaining: Optional[float] = None): ...

class DocumentIngestionError(BaseAppException):
    # status_code = 422 Unprocessable Entity
    def __init__(self, document_id: Any, reason: str): ...
```

### M2 (T-302) Circuit Breaker & AI Client Contract
```python
# backend/app/shared/clients/circuit_breaker.py
class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, service_name: str = "GeminiService", failure_threshold: int = 3, recovery_timeout: float = 30.0): ...
    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T: ...
    def record_success(self) -> None: ...
    def record_failure(self, exc: Exception) -> None: ...
    def reset(self) -> None: ...

# backend/app/shared/clients/gemini_client.py
class GeminiClient:
    circuit_breaker: CircuitBreaker
    async def generate_embedding(self, text: str) -> List[float]: ...  # 768-dim vector, retried with jitter
    async def generate_chat_response(self, prompt: str, system_instruction: Optional[str] = None) -> str: ...
```

### M3 (T-301) Chunker, Embedder & Repository Contract
```python
# backend/app/shared/rag_engine/chunker.py
class ChunkResult(BaseModel):
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any]
    start_char: int
    end_char: int

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: Optional[List[str]] = None): ...
    def split_text(self, text: str) -> List[str]: ...
    def split_text_with_metadata(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]: ...
    def split_markdown(self, markdown_text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]: ...

# backend/app/modules/documents/interfaces/i_chunk_repository.py
class IChunkRepository(IBaseRepository[DocumentChunk], ABC):
    @abstractmethod
    async def create_chunks(self, chunks: Sequence[DocumentChunk]) -> Sequence[DocumentChunk]: ...
```

### M4 (T-304) Ingestion Tasks & Controller Contract
```python
# backend/app/shared/tasks/ingestion_tasks.py
async def process_document_background_task(
    document_id: uuid.UUID,
    department: str,
    raw_content: Optional[str] = None
) -> None:
    """Uses independent AsyncSessionLocal(), transitions QUEUED -> PROCESSING -> SUCCESS/FAILED."""

# backend/app/modules/documents/controllers/documents_controller.py
@router.post("/", response_model=APIResponse[DocumentDetailResponse], status_code=status.HTTP_201_CREATED)
async def create_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: IDocumentService = Depends(get_document_service),
) -> APIResponse[DocumentDetailResponse]:
    """Returns immediately in < 100ms, enqueuing background ingestion."""
```

## Code Layout & Write Boundaries
- `backend/app/core/exceptions.py` -> Owned by M1 Worker
- `backend/app/shared/clients/` -> Owned by M2 Worker
- `backend/app/shared/rag_engine/` & `backend/app/modules/documents/repositories/` -> Owned by M3 Worker
- `backend/app/shared/tasks/` & `backend/app/modules/documents/controllers/` -> Owned by M4 Worker
- `backend/tests/` -> Owned by QA Writers / Testers
