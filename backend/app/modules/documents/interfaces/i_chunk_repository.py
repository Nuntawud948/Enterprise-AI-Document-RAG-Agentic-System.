from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple
from app.modules.documents.models.document import DocumentChunk
from app.shared.repositories.i_base_repository import IBaseRepository


class IChunkRepository(IBaseRepository[DocumentChunk]):
    """Contract for DocumentChunk data persistence and pgvector similarity retrieval."""

    @abstractmethod
    async def create_chunks(self, chunks: Sequence[DocumentChunk]) -> List[DocumentChunk]:
        """Bulk-insert persisting all chunks in a single transaction."""
        pass

    @abstractmethod
    async def list_by_document(self, document_id: Any) -> Sequence[DocumentChunk]:
        """List all chunks belonging to a document ordered by chunk_index."""
        pass

    @abstractmethod
    async def search_similar_chunks(
        self,
        embedding: List[float],
        department: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        allowed_clearances: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform approximate nearest neighbor (ANN) vector similarity search using HNSW index.
        Filters strictly by department for multi-tenant isolation.
        Filters by allowed_clearances for data leak prevention.
        Returns a list of tuples containing (DocumentChunk, similarity_score).
        """
        pass

    @abstractmethod
    async def delete_by_document(self, document_id: Any) -> int:
        """Delete all chunks belonging to a document."""
        pass
