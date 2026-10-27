import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.modules.auth.models.user import User
from app.shared.models.base_entity import DepartmentScopedEntity


class Document(DepartmentScopedEntity):
    """
    Enterprise Document metadata entity.
    Scoped by 'department' for Multi-Tenancy row-level isolation.
    """
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)  # pdf, docx, txt
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="QUEUED",
        nullable=False,
        comment="Ingestion status: QUEUED | PROCESSING | COMPLETED | FAILED"
    )
    uploader_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    classification_level: Mapped[str] = mapped_column(
        String(50),
        default="INTERNAL",
        nullable=False,
        index=True,
        comment="Security classification level: PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED"
    )

    # Relationship to Uploader (lazy="raise" enforces explicit loading in repository)
    uploader: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[uploader_id],
        lazy="raise"
    )

    # Relationship to Chunks
    # [CRITICAL WATCH-OUT FIX]: lazy="raise" prevents accidental N+1 and massive memory loads
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="raise"
    )


class DocumentChunk(DepartmentScopedEntity):
    """
    Individual chunk of text with its vector embedding.
    Notice 'department' is denormalized directly on chunks for fast, isolated vector filtering.
    """
    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 768-dimensional vector embedding for Google Gemini (text-embedding-004)
    embedding = mapped_column(Vector(768), nullable=True)

    # Metadata dictionary for fine-grained retrieval filters
    chunk_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Relationship back to parent document
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_document_chunks_department_document",
            "department",
            "document_id",
        ),
    )
