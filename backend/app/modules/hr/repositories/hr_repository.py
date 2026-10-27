from typing import Optional, Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.hr.interfaces.i_hr_repository import IHRRepository
from app.modules.hr.models.hr_policy import HRPolicy
from app.shared.repositories.base_repository import BaseRepository


class HRRepository(BaseRepository[HRPolicy], IHRRepository):
    """SQLAlchemy implementation for HR policies."""

    def __init__(self, session: AsyncSession):
        super().__init__(model_class=HRPolicy, session=session)

    async def get_by_code(self, policy_code: str) -> Optional[HRPolicy]:
        stmt = select(HRPolicy).where(HRPolicy.policy_code == policy_code)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_category(self, category: Optional[str] = None) -> Sequence[HRPolicy]:
        stmt = select(HRPolicy)
        if category:
            stmt = stmt.where(HRPolicy.category == category)
        result = await self.session.execute(stmt)
        return result.scalars().all()
