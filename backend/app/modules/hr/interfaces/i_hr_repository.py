from abc import abstractmethod
from typing import Optional, Sequence
from app.modules.hr.models.hr_policy import HRPolicy
from app.shared.repositories.i_base_repository import IBaseRepository


class IHRRepository(IBaseRepository[HRPolicy]):
    """Contract for HR Policy data access."""

    @abstractmethod
    async def get_by_code(self, policy_code: str) -> Optional[HRPolicy]:
        """Fetch policy by unique code."""
        pass

    @abstractmethod
    async def list_by_category(self, category: Optional[str] = None) -> Sequence[HRPolicy]:
        """List policies filtered by category."""
        pass
