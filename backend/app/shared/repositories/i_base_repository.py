from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Sequence, TypeVar

T = TypeVar("T")


class IBaseRepository(ABC, Generic[T]):
    """
    Contract interface for Generic CRUD Data Access operations.
    Supports Dependency Inversion & Mocking for Unit Tests.
    """

    @abstractmethod
    async def get_by_id(self, entity_id: Any) -> Optional[T]:
        """Fetch a single entity by primary key."""
        pass

    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> Sequence[T]:
        """List entities with basic offset pagination."""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Persist a new entity to the database."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: Any) -> bool:
        """Delete an entity by primary key."""
        pass
