import logging
import uuid
import pytest
from fastapi import APIRouter, status
from fastapi.testclient import TestClient

from app.core.exceptions import (
    BaseAppException,
    EntityNotFoundException,
    EntityNotFoundError,
    VectorServiceError,
    QuotaExceededError,
    CircuitBreakerOpenError,
    DocumentIngestionError,
)
from app.main import app

# Create dedicated test router for TICKET-103 resiliency verification
resiliency_router = APIRouter(prefix="/test-resiliency", tags=["Test Error Resiliency"])


@resiliency_router.get("/vector-error")
def trigger_vector_service_error():
    raise VectorServiceError("pgvector connection timeout occurred.")


@resiliency_router.get("/quota-error")
def trigger_quota_exceeded_error():
    raise QuotaExceededError("Gemini API rate limit reached.", retry_after=60)


@resiliency_router.get("/circuit-breaker-open")
def trigger_circuit_breaker_open_error():
    raise CircuitBreakerOpenError("GeminiEmbeddingService", recovery_time_remaining=25.4)


@resiliency_router.get("/circuit-breaker-no-retry")
def trigger_circuit_breaker_no_retry_error():
    raise CircuitBreakerOpenError("ExternalVectorService", recovery_time_remaining=None)


@resiliency_router.get("/document-ingestion-error")
def trigger_document_ingestion_error():
    raise DocumentIngestionError("doc-sample-101", "Corrupted PDF binary header.")


@resiliency_router.get("/entity-not-found-error")
def trigger_entity_not_found_error():
    raise EntityNotFoundError("DepartmentPolicy", "pol-uuid-999")


@resiliency_router.get("/unhandled-system-error")
def trigger_unhandled_system_error():
    # Simulate an unhandled bug or deep database driver error leaking sensitive details
    raise RuntimeError(
        "CRITICAL: Failed query with connection string postgresql://app_user:super_secret_pwd_99@prod-db:5432/core"
    )


# Register test routes on main app
app.include_router(resiliency_router)

# Note: For testing unhandled 500 exceptions, raise_server_exceptions=False prevents TestClient
# from re-raising unhandled exceptions, allowing Starlette's ServerErrorMiddleware to execute.
client = TestClient(app, raise_server_exceptions=False)


# ============================================================================
# 1. Unit Tests: Exception Instantiation & Properties
# ============================================================================

def test_entity_not_found_alias():
    """Verify EntityNotFoundError is an exact alias for EntityNotFoundException with HTTP 404."""
    assert EntityNotFoundError is EntityNotFoundException
    exc = EntityNotFoundError("DocumentChunk", "chunk-uuid-123")
    assert isinstance(exc, BaseAppException)
    assert exc.status_code == status.HTTP_404_NOT_FOUND
    assert "DocumentChunk with identifier 'chunk-uuid-123' was not found." in exc.message
    assert exc.errors == ["DocumentChunk not found."]


def test_vector_service_error_defaults_and_custom():
    """Verify VectorServiceError defaults to HTTP 502 and supports custom messages and errors."""
    default_exc = VectorServiceError()
    assert default_exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert default_exc.message == "Vector embedding service failed."
    assert default_exc.errors == ["Vector embedding service failed."]

    custom_exc = VectorServiceError(
        message="Vector database unreachable.",
        errors=["pgvector connection refused at host 10.0.0.5", "timeout 5000ms"]
    )
    assert custom_exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert custom_exc.message == "Vector database unreachable."
    assert len(custom_exc.errors) == 2
    assert "pgvector connection refused at host 10.0.0.5" in custom_exc.errors


def test_quota_exceeded_error_with_and_without_retry():
    """Verify QuotaExceededError maps to HTTP 429 and handles optional retry_after."""
    exc_without_retry = QuotaExceededError("Rate limit exceeded.")
    assert exc_without_retry.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc_without_retry.retry_after is None
    assert exc_without_retry.errors == ["Rate limit exceeded."]

    exc_with_retry = QuotaExceededError("Token quota exhausted.", retry_after=45)
    assert exc_with_retry.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc_with_retry.retry_after == 45
    assert exc_with_retry.errors == [
        "Token quota exhausted.",
        "Retry after 45 seconds."
    ]


def test_circuit_breaker_open_error_formatting():
    """Verify CircuitBreakerOpenError maps to HTTP 503 and calculates retry_after from recovery time."""
    exc = CircuitBreakerOpenError("GeminiEmbedding", recovery_time_remaining=15.7)
    assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc.service_name == "GeminiEmbedding"
    assert exc.recovery_time_remaining == 15.7
    assert exc.retry_after == 15
    assert any("Circuit breaker for service 'GeminiEmbedding' is OPEN" in err for err in exc.errors)
    assert any("Recovery estimated in 15.7s" in err for err in exc.errors)

    # Edge case: sub-second recovery time defaults to minimum 1s retry_after
    exc_sub_second = CircuitBreakerOpenError("ExternalService", recovery_time_remaining=0.3)
    assert exc_sub_second.retry_after == 1

    # Edge case: no recovery time remaining provided
    exc_no_recovery = CircuitBreakerOpenError("ExternalService", recovery_time_remaining=None)
    assert exc_no_recovery.retry_after is None
    assert exc_no_recovery.recovery_time_remaining is None


def test_document_ingestion_error_properties():
    """Verify DocumentIngestionError maps to HTTP 422 and retains document_id and reason."""
    exc = DocumentIngestionError("doc-999", "Failed to extract text from encrypted PDF.")
    assert exc.status_code == 422
    assert exc.document_id == "doc-999"
    assert exc.reason == "Failed to extract text from encrypted PDF."
    assert exc.errors == ["Ingestion failed for document 'doc-999': Failed to extract text from encrypted PDF."]

    custom_errors = ["Header block corrupted", "CRC check failed"]
    exc_custom = DocumentIngestionError("doc-999", "Corrupted file", errors=custom_errors)
    assert exc_custom.errors == custom_errors


# ============================================================================
# 2. Integration Tests: Live HTTP Routes & Standardized Envelopes
# ============================================================================

def test_endpoint_vector_service_error():
    """Verify VectorServiceError returns HTTP 502 with standard APIResponse envelope."""
    response = client.get("/test-resiliency/vector-error")
    assert response.status_code == status.HTTP_502_BAD_GATEWAY

    body = response.json()
    assert body["code"] == 502
    assert body["status"] == "error"
    assert body["data"] is None
    assert "pgvector connection timeout occurred." in body["errors"]
    assert "request_id" in body["meta"]
    assert "timestamp" in body["meta"]


def test_endpoint_quota_exceeded_error_retry_after_header():
    """Verify QuotaExceededError returns HTTP 429 and injects standard Retry-After header."""
    response = client.get("/test-resiliency/quota-error")
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert response.headers.get("Retry-After") == "60"

    body = response.json()
    assert body["code"] == 429
    assert body["status"] == "error"
    assert body["data"] is None
    assert "Gemini API rate limit reached." in body["errors"]
    assert "Retry after 60 seconds." in body["errors"]


def test_endpoint_circuit_breaker_open_retry_after_header():
    """Verify CircuitBreakerOpenError returns HTTP 503 and injects standard Retry-After header."""
    response = client.get("/test-resiliency/circuit-breaker-open")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.headers.get("Retry-After") == "25"

    body = response.json()
    assert body["code"] == 503
    assert body["status"] == "error"
    assert body["data"] is None
    assert any("Circuit breaker for service 'GeminiEmbeddingService' is OPEN" in err for err in body["errors"])


def test_endpoint_circuit_breaker_no_retry_after():
    """Verify CircuitBreakerOpenError with None recovery_time omits Retry-After header."""
    response = client.get("/test-resiliency/circuit-breaker-no-retry")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Retry-After" not in response.headers

    body = response.json()
    assert body["code"] == 503
    assert body["status"] == "error"


def test_endpoint_document_ingestion_error():
    """Verify DocumentIngestionError returns HTTP 422 with standard envelope."""
    response = client.get("/test-resiliency/document-ingestion-error")
    assert response.status_code == 422

    body = response.json()
    assert body["code"] == 422
    assert body["status"] == "error"
    assert "Ingestion failed for document 'doc-sample-101': Corrupted PDF binary header." in body["errors"]


def test_endpoint_entity_not_found_error():
    """Verify EntityNotFoundError returns HTTP 404 with standard envelope."""
    response = client.get("/test-resiliency/entity-not-found-error")
    assert response.status_code == status.HTTP_404_NOT_FOUND

    body = response.json()
    assert body["code"] == 404
    assert body["status"] == "error"
    assert body["errors"] == ["DepartmentPolicy not found."]


# ============================================================================
# 3. Integration Tests: Sanitization & Masking of Unhandled 500 Exceptions
# ============================================================================

def test_endpoint_unhandled_500_masks_stack_trace_and_internal_secrets():
    """
    Verify that unhandled server exceptions (HTTP 500) do NOT leak stack traces,
    database passwords, file paths, or raw exception names to HTTP response bodies,
    while correctly preserving the X-Request-ID header and error envelope.
    """
    custom_trace_id = "test-unhandled-trace-500-abc"
    response = client.get(
        "/test-resiliency/unhandled-system-error",
        headers={"X-Request-ID": custom_trace_id}
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Header check: X-Request-ID must be propagated even on unhandled 500s
    assert response.headers.get("X-Request-ID") == custom_trace_id

    # Envelope check
    body = response.json()
    assert body["code"] == 500
    assert body["status"] == "error"
    assert body["data"] is None
    assert body["meta"]["request_id"] == custom_trace_id
    assert "timestamp" in body["meta"]
    assert body["errors"] == [
        "An unexpected internal server error occurred. Please contact administrator with request_id."
    ]

    # Security check: Zero leak of internal secrets, file paths, or exception types
    response_text = response.text
    assert "super_secret_pwd_99" not in response_text
    assert "prod-db" not in response_text
    assert "postgresql://" not in response_text
    assert "RuntimeError" not in response_text
    assert "Traceback" not in response_text
    assert ".py" not in response_text


def test_unhandled_500_generates_request_id_when_omitted():
    """Verify that an unhandled 500 automatically generates a UUID request_id and sets X-Request-ID header."""
    response = client.get("/test-resiliency/unhandled-system-error")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    header_req_id = response.headers.get("X-Request-ID")
    assert header_req_id is not None
    # Verify it is a valid UUID
    parsed_uuid = uuid.UUID(header_req_id)
    assert str(parsed_uuid) == header_req_id

    body = response.json()
    assert body["meta"]["request_id"] == header_req_id


def test_unhandled_500_logs_full_traceback_with_request_id(caplog):
    """
    Verify that while clients receive sanitized responses, internal system logs
    capture the complete error traceback and sensitive details alongside request_id.
    """
    trace_id = "trace-log-audit-999"
    with caplog.at_level(logging.ERROR):
        response = client.get(
            "/test-resiliency/unhandled-system-error",
            headers={"X-Request-ID": trace_id}
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Check logger output
    assert f"Unhandled system error [request_id={trace_id}]" in caplog.text
    assert "super_secret_pwd_99" in caplog.text
    assert "RuntimeError" in caplog.text


# ============================================================================
# 4. Integration Tests: Request ID Correlation Consistency Across All Routes
# ============================================================================

@pytest.mark.parametrize(
    "route_path,expected_status",
    [
        ("/test-resiliency/vector-error", 502),
        ("/test-resiliency/quota-error", 429),
        ("/test-resiliency/circuit-breaker-open", 503),
        ("/test-resiliency/circuit-breaker-no-retry", 503),
        ("/test-resiliency/document-ingestion-error", 422),
        ("/test-resiliency/entity-not-found-error", 404),
        ("/test-resiliency/unhandled-system-error", 500),
    ]
)
def test_request_id_correlation_consistency(route_path, expected_status):
    """
    Verify that every error type (4xx, 5xx, and unhandled 500) preserves the client's
    correlation X-Request-ID in both the response HTTP header and the APIResponse meta envelope.
    """
    test_req_id = f"corr-check-{uuid.uuid4()}"
    response = client.get(route_path, headers={"X-Request-ID": test_req_id})

    assert response.status_code == expected_status
    assert response.headers.get("X-Request-ID") == test_req_id

    body = response.json()
    assert body["meta"]["request_id"] == test_req_id
    assert body["status"] == "error"
