import uuid
import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import ConflictException, EntityNotFoundException
from app.modules.hr.dtos.requests.hr_requests import CreateHRPolicyRequest
from app.modules.hr.interfaces.i_hr_repository import IHRRepository
from app.modules.hr.models.hr_policy import HRPolicy
from app.modules.hr.services.hr_service import HRService


@pytest.mark.asyncio
async def test_hr_service_create_policy_success():
    mock_repo = AsyncMock(spec=IHRRepository)
    mock_repo.get_by_code.return_value = None

    async def mock_create(policy: HRPolicy):
        policy.id = uuid.uuid4()
        return policy

    mock_repo.create.side_effect = mock_create

    service = HRService(hr_repo=mock_repo)
    req = CreateHRPolicyRequest(
        policy_code="HR-LEAVE-01",
        title="Annual Leave Policy",
        category="Leave",
        summary="Policy regarding annual paid leave."
    )
    result = await service.create_policy(req)

    assert result.policy_code == "HR-LEAVE-01"
    assert result.title == "Annual Leave Policy"
    mock_repo.get_by_code.assert_awaited_once_with("HR-LEAVE-01")
    mock_repo.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_hr_service_create_duplicate_conflict():
    mock_repo = AsyncMock(spec=IHRRepository)
    mock_repo.get_by_code.return_value = HRPolicy(
        policy_code="HR-LEAVE-01",
        title="Existing",
        category="Leave",
        summary="Existing policy text long enough"
    )

    service = HRService(hr_repo=mock_repo)
    req = CreateHRPolicyRequest(
        policy_code="HR-LEAVE-01",
        title="Duplicate",
        category="Leave",
        summary="Duplicate policy text long enough"
    )

    with pytest.raises(ConflictException) as exc_info:
        await service.create_policy(req)

    assert "already exists" in str(exc_info.value)
    mock_repo.create.assert_not_called()


@pytest.mark.asyncio
async def test_hr_service_get_by_code_not_found():
    mock_repo = AsyncMock(spec=IHRRepository)
    mock_repo.get_by_code.return_value = None

    service = HRService(hr_repo=mock_repo)

    with pytest.raises(EntityNotFoundException):
        await service.get_by_code("NON-EXISTENT")
