from typing import Any, Generic, Optional, Sequence, Type, TypeVar
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.repositories.i_base_repository import IBaseRepository

T = TypeVar("T")


class BaseRepository(IBaseRepository[T], Generic[T]):
    """
    SQLAlchemy 2.0 Async Generic Repository Implementation.
    """

    def __init__(self, model_class: Type[T], session: AsyncSession):
        self.model_class = model_class
        self.session = session

    async def get_by_id(self, entity_id: Any) -> Optional[T]:
        result = await self.session.get(self.model_class, entity_id)
        return result

    async def list_all(self, skip: int = 0, limit: int = 100) -> Sequence[T]:
        stmt = select(self.model_class).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity: T) -> T:
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: Any) -> bool:
        entity = await self.get_by_id(entity_id)
        if not entity:
            return False
        await self.session.delete(entity)
        await self.session.flush()
        return True
