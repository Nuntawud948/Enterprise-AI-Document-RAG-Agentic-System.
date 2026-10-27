from typing import Any, Optional, Sequence
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.modules.documents.interfaces.i_document_repository import IDocumentRepository
from app.modules.documents.models.document import Document
from app.shared.repositories.base_repository import BaseRepository


class DocumentRepository(BaseRepository[Document], IDocumentRepository):
    """
    SQLAlchemy implementation of IDocumentRepository enforcing Department boundaries.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(model_class=Document, session=session)

    async def get_by_id_with_chunks(self, document_id: Any) -> Optional[Document]:
        """
        Explicit eager load of chunks and uploader on-demand.
        Prevents N+1 queries while keeping list queries lightweight.
        """
        stmt = (
            select(Document)
            .options(
                selectinload(Document.chunks),
                joinedload(Document.uploader)
            )
            .where(Document.id == document_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_department(
        self,
        department: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Sequence[Document]:
        stmt = select(Document).where(Document.department == department)
        if status:
            stmt = stmt.where(Document.status == status)
        stmt = stmt.order_by(Document.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_department(
        self,
        department: str,
        status: Optional[str] = None
    ) -> int:
        stmt = select(func.count(Document.id)).where(Document.department == department)
        if status:
            stmt = stmt.where(Document.status == status)
        result = await self.session.execute(stmt)
        return result.scalar() or 0
