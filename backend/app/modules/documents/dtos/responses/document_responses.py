import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkResponse(BaseModel):
    """Response DTO for individual document chunks."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    department: str


class DocumentDetailResponse(BaseModel):
    """Response DTO for full document details including its chunks count."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    file_name: str
    file_type: str
    file_size_bytes: int
    status: str
    department: str
    classification_level: str = "INTERNAL"
    uploader_id: uuid.UUID
    chunks_count: int = Field(default=0, description="Total chunks extracted")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VectorSearchResultItem(BaseModel):
    """Item DTO for vector search results."""
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    department: str
    similarity_score: float


class VectorSearchResponse(BaseModel):
    """Response DTO for vector search endpoint."""
    query_department: str
    clearance_level: str
    total_results: int
    results: List[VectorSearchResultItem]
