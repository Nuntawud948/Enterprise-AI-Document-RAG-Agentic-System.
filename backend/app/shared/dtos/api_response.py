from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    """Metadata attached to every standardized API response for observability & tracing."""
    request_id: Optional[str] = Field(default=None, description="Unique correlation ID of the request")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of response generation"
    )


class APIResponse(BaseModel, Generic[T]):
    """
    Standardized API Response Wrapper Envelope (Enterprise Contract).
    Ensures every response across all departments has identical structure.
    """
    code: int = Field(default=200, description="HTTP status code integer")
    status: str = Field(default="success", description="Status indicator: 'success' | 'error'")
    data: Optional[T] = Field(default=None, description="Payload data returned to the caller")
    errors: Optional[List[str]] = Field(default=None, description="List of human-readable error messages if any")
    meta: ResponseMeta = Field(default_factory=ResponseMeta, description="Response metadata")

    @classmethod
    def success_response(
        cls,
        data: Optional[T] = None,
        code: int = 200,
        request_id: Optional[str] = None
    ) -> "APIResponse[T]":
        """Factory method for consistent success responses."""
        return cls(
            code=code,
            status="success",
            data=data,
            errors=None,
            meta=ResponseMeta(request_id=request_id)
        )

    @classmethod
    def error_response(
        cls,
        errors: List[str],
        code: int = 400,
        request_id: Optional[str] = None
    ) -> "APIResponse[None]":
        """Factory method for consistent error responses."""
        return cls(
            code=code,
            status="error",
            data=None,
            errors=errors,
            meta=ResponseMeta(request_id=request_id)
        )
