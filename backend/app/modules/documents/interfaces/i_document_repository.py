from abc import abstractmethod
from typing import Any, Optional, Sequence
from app.modules.documents.models.document import Document
from app.shared.repositories.i_base_repository import IBaseRepository


class IDocumentRepository(IBaseRepository[Document]):
    """Contract for Document data persistence scoped by Department."""

    @abstractmethod
    async def get_by_id_with_chunks(self, document_id: Any) -> Optional[Document]:
        """
        Retrieve a single document with chunks and uploader eagerly loaded.
        Prevents N+1 queries using explicit selectinload / joinedload.
        """
        pass

    @abstractmethod
    async def list_by_department(
        self,
        department: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Sequence[Document]:
        """List documents filtered strictly by department (without eager loading chunks to prevent payload bloat)."""
        pass

    @abstractmethod
    async def count_by_department(
        self,
        department: str,
        status: Optional[str] = None
    ) -> int:
        """Count total documents in department matching filter."""
        pass
