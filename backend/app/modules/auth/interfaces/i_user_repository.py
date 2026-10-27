from abc import abstractmethod
from typing import Optional
from app.modules.auth.models.user import User
from app.shared.repositories.i_base_repository import IBaseRepository


class IUserRepository(IBaseRepository[User]):
    """Contract for User entity data persistence."""

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Fetch user by unique email address."""
        pass
