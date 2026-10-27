from typing import Generic, List, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Common request DTO for paginated queries across all department modules."""
    page: int = Field(default=1, ge=1, description="Page number starting from 1")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")


class PagedResult(BaseModel, Generic[T]):
    """Standard pagination wrapper inside APIResponse data payload."""
    items: List[T] = Field(default_factory=list, description="List of items for current page")
    total_count: int = Field(default=0, ge=0, description="Total number of items matching filter")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=20, ge=1, description="Page size")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")

    @classmethod
    def create(cls, items: List[T], total_count: int, page: int, page_size: int) -> "PagedResult[T]":
        total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            items=items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
