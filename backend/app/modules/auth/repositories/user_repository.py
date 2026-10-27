from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.interfaces.i_user_repository import IUserRepository
from app.modules.auth.models.user import User
from app.shared.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User], IUserRepository):
    """
    SQLAlchemy implementation of IUserRepository.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(model_class=User, session=session)

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
