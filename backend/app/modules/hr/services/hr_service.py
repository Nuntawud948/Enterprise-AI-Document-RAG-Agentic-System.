from typing import List, Optional
from app.core.exceptions import ConflictException, EntityNotFoundException
from app.modules.hr.dtos.requests.hr_requests import CreateHRPolicyRequest
from app.modules.hr.dtos.responses.hr_responses import HRPolicyResponse
from app.modules.hr.interfaces.i_hr_repository import IHRRepository
from app.modules.hr.interfaces.i_hr_service import IHRService
from app.modules.hr.models.hr_policy import HRPolicy


class HRService(IHRService):
    """Business logic for HR policies."""

    def __init__(self, hr_repo: IHRRepository):
        self.hr_repo = hr_repo

    async def create_policy(self, request: CreateHRPolicyRequest) -> HRPolicyResponse:
        existing = await self.hr_repo.get_by_code(request.policy_code)
        if existing:
            raise ConflictException(f"HR Policy with code '{request.policy_code}' already exists.")

        policy = HRPolicy(
            policy_code=request.policy_code.upper(),
            title=request.title,
            category=request.category,
            summary=request.summary,
            department="HR"
        )
        created = await self.hr_repo.create(policy)
        return HRPolicyResponse.model_validate(created)

    async def get_by_code(self, policy_code: str) -> HRPolicyResponse:
        policy = await self.hr_repo.get_by_code(policy_code.upper())
        if not policy:
            raise EntityNotFoundException("HRPolicy", policy_code)
        return HRPolicyResponse.model_validate(policy)

    async def list_policies(self, category: Optional[str] = None) -> List[HRPolicyResponse]:
        policies = await self.hr_repo.list_by_category(category)
        return [HRPolicyResponse.model_validate(p) for p in policies]
