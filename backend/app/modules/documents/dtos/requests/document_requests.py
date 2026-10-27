from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    """Request DTO to initiate document registration/upload."""
    title: str = Field(..., min_length=2, max_length=255, description="Document display title")
    file_name: str = Field(..., min_length=1, max_length=255, description="Original filename with extension")
    file_type: str = Field(..., description="File MIME or extension: pdf, docx, txt, md")
    file_size_bytes: int = Field(..., gt=0, description="Size in bytes")
    classification_level: Optional[str] = Field(
        default="INTERNAL",
        description="Security classification: PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED"
    )
    content: Optional[str] = Field(
        default=None,
        description="Direct text content for ingestion"
    )


# Backward-compatible specification alias
DocumentUploadRequest = DocumentCreateRequest


class DocumentFilterRequest(BaseModel):
    """Request DTO to filter documents by department and status."""
    status: Optional[str] = Field(
        default=None,
        description="Optional status filter: QUEUED, PROCESSING, SUCCESS, COMPLETED, FAILED"
    )
    classification_level: Optional[str] = Field(default=None, description="Optional classification level filter")


class VectorSearchRequest(BaseModel):
    """Request DTO for semantic similarity vector search."""
    query_vector: List[float] = Field(..., description="768-dim query vector")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top matches to return")
    score_threshold: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0.0 - 1.0)"
    )
