import logging
from typing import Any, List, Optional
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.shared.dtos.api_response import APIResponse

logger = logging.getLogger(__name__)


class BaseAppException(Exception):
    """Base class for all domain/application specific exceptions."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.errors = errors or [message]


class EntityNotFoundException(BaseAppException):
    def __init__(self, entity_name: str, identifier: Any):
        super().__init__(
            message=f"{entity_name} with identifier '{identifier}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            errors=[f"{entity_name} not found."]
        )


# Alias for sprint naming convention compliance
EntityNotFoundError = EntityNotFoundException


class ValidationException(BaseAppException):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            errors=errors or [message]
        )


class UnauthorizedException(BaseAppException):
    def __init__(self, message: str = "Authentication credentials were not provided or invalid."):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            errors=[message]
        )


class ForbiddenException(BaseAppException):
    def __init__(self, message: str = "You do not have permission to perform this action."):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            errors=[message]
        )


class ConflictException(BaseAppException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            errors=[message]
        )



class VectorServiceError(BaseAppException):
    """
    Raised when vector embedding, similarity search, or pgvector operations fail.
    Maps to HTTP 502 Bad Gateway.
    """
    def __init__(
        self,
        message: str = "Vector embedding service failed.",
        errors: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            errors=errors or [message]
        )


class QuotaExceededError(BaseAppException):
    """
    Raised when AI API token quotas or rate limits are exhausted (HTTP 429).
    Optionally specifies retry_after in seconds.
    """
    def __init__(
        self,
        message: str = "AI service quota or rate limit exceeded.",
        retry_after: Optional[int] = None
    ):
        errors = [message]
        if retry_after is not None:
            errors.append(f"Retry after {retry_after} seconds.")
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            errors=errors
        )
        self.retry_after = retry_after


class CircuitBreakerOpenError(BaseAppException):
    """
    Raised when an external call is attempted while the circuit breaker is OPEN (HTTP 503).
    Prevents cascading downstream failure and fast-fails requests.
    """
    def __init__(
        self,
        service_name: str,
        recovery_time_remaining: Optional[float] = None
    ):
        detail = f"Circuit breaker for service '{service_name}' is OPEN. Fast failing request."
        errors = [detail]
        if recovery_time_remaining is not None:
            errors.append(f"Recovery estimated in {recovery_time_remaining:.1f}s.")
        super().__init__(
            message=detail,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            errors=errors
        )
        self.service_name = service_name
        self.recovery_time_remaining = recovery_time_remaining
        self.retry_after = max(1, int(recovery_time_remaining)) if recovery_time_remaining is not None else None


class DocumentIngestionError(BaseAppException):
    """
    Raised when document parsing, validation, chunking, or background ingestion fails.
    Maps to HTTP 422 Unprocessable Content / Entity.
    """
    def __init__(
        self,
        document_id: Any,
        reason: str,
        errors: Optional[List[str]] = None
    ):
        detail = f"Ingestion failed for document '{document_id}': {reason}"
        super().__init__(
            message=detail,
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            errors=errors or [detail]
        )
        self.document_id = document_id
        self.reason = reason


# Global Exception Handlers returning APIResponse Envelope

async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    response_payload = APIResponse.error_response(
        errors=exc.errors,
        code=exc.status_code,
        request_id=request_id
    )
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=exc.status_code,
        content=response_payload.model_dump(),
        headers=headers if headers else None
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    error_messages = [f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()]
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
    response_payload = APIResponse.error_response(
        errors=error_messages,
        code=status_code,
        request_id=request_id
    )
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content=response_payload.model_dump(),
        headers=headers
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    detail = exc.detail if isinstance(exc.detail, list) else [str(exc.detail)]
    response_payload = APIResponse.error_response(
        errors=detail,
        code=exc.status_code,
        request_id=request_id
    )
    headers = dict(exc.headers) if exc.headers else {}
    if request_id and "X-Request-ID" not in headers:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=exc.status_code,
        content=response_payload.model_dump(),
        headers=headers if headers else None
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled system error [request_id={request_id}]: {str(exc)}")
    response_payload = APIResponse.error_response(
        errors=["An unexpected internal server error occurred. Please contact administrator with request_id."],
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id
    )
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_payload.model_dump(),
        headers=headers
    )

