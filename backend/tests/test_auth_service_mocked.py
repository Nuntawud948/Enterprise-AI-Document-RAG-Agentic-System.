import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import ConflictException
from app.modules.auth.dtos.requests.auth_requests import RegisterRequest
from app.modules.auth.interfaces.i_user_repository import IUserRepository
from app.modules.auth.models.user import User, UserRole
from app.modules.auth.services.auth_service import AuthService


@pytest.mark.asyncio
async def test_auth_service_register_success_mocked():
    """
    Demonstrates clean OOP Interface Mocking:
    Tests AuthService business logic without starting PostgreSQL or running DB queries.
    """
    # 1. Arrange: Create mock repository implementing IUserRepository
    mock_user_repo = AsyncMock(spec=IUserRepository)
    mock_user_repo.get_by_email.return_value = None  # No existing user
    
    # Mock create returning populated entity
    async def mock_create(user_entity: User):
        user_entity.id = "11111111-1111-1111-1111-111111111111"
        return user_entity
        
    mock_user_repo.create.side_effect = mock_create
    
    service = AuthService(user_repo=mock_user_repo)
    
    request = RegisterRequest(
        email="developer@enterprise.com",
        password="SecurePassword123!",
        full_name="Enterprise Dev",
        department="ENGINEERING"
    )
    
    # 2. Act
    result = await service.register(request)
    
    # 3. Assert
    assert result.email == "developer@enterprise.com"
    assert result.department == "ENGINEERING"
    assert result.role == "MEMBER"
    mock_user_repo.get_by_email.assert_awaited_once_with("developer@enterprise.com")
    mock_user_repo.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_service_register_conflict_throws_exception():
    """
    Verify business logic throws ConflictException when email is already registered.
    """
    mock_user_repo = AsyncMock(spec=IUserRepository)
    existing_user = User(email="duplicate@enterprise.com", department="HR")
    mock_user_repo.get_by_email.return_value = existing_user
    
    service = AuthService(user_repo=mock_user_repo)
    
    request = RegisterRequest(
        email="duplicate@enterprise.com",
        password="SecurePassword123!",
        full_name="Duplicate User",
        department="HR"
    )
    
    with pytest.raises(ConflictException) as exc_info:
        await service.register(request)
        
    assert "already exists" in str(exc_info.value)
    mock_user_repo.create.assert_not_called()
