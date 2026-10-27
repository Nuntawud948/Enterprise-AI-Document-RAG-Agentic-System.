from typing import Any, List, Optional, Sequence, Tuple
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.documents.interfaces.i_chunk_repository import IChunkRepository
from app.modules.documents.models.document import Document, DocumentChunk
from app.shared.repositories.base_repository import BaseRepository


class ChunkRepository(BaseRepository[DocumentChunk], IChunkRepository):
    """
    SQLAlchemy & pgvector implementation of IChunkRepository.
    Enforces Department isolation, Classification Clearance, and HNSW cosine similarity search.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(model_class=DocumentChunk, session=session)

    async def create_chunks(self, chunks: Sequence[DocumentChunk]) -> List[DocumentChunk]:
        """
        Bulk-insert persisting all chunks in a single transaction.
        Converts sequence to list, guards empty, and uses session.add_all and flush.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return []
        self.session.add_all(chunk_list)
        await self.session.flush()
        return chunk_list

    async def list_by_document(self, document_id: Any) -> Sequence[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def search_similar_chunks(
        self,
        embedding: List[float],
        department: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        allowed_clearances: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search top_k nearest chunks with strict multi-tenant and metadata isolation.
        - Isolates strictly by department (Row-Level Security).
        - Isolates strictly by security clearance level (Data Leak Prevention).
        - Computes cosine distance via pgvector's <=> operator.
        """
        distance = DocumentChunk.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("similarity_score")

        stmt = (
            select(DocumentChunk, similarity)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.department == department)
        )

        # Mandatory Data Leak Prevention: filter only documents within user's clearance
        if allowed_clearances is not None:
            stmt = stmt.where(Document.classification_level.in_(allowed_clearances))

        # Filter by minimum similarity score
        if score_threshold is not None:
            stmt = stmt.where(distance <= (1.0 - score_threshold))

        stmt = stmt.order_by(distance.asc()).limit(top_k)

        result = await self.session.execute(stmt)
        rows = result.all()
        return [(row[0], float(row[1])) for row in rows]

    async def delete_by_document(self, document_id: Any) -> int:
        stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        result = await self.session.execute(stmt)
        return result.rowcount or 0
