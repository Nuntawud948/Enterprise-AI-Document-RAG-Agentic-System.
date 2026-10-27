from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.exceptions import ForbiddenException
from app.core.security import AuthenticatedUser, get_current_user, require_roles
from app.modules.auth.models.user import UserRole
from app.modules.hr.dtos.requests.hr_requests import CreateHRPolicyRequest
from app.modules.hr.dtos.responses.hr_responses import HRPolicyResponse
from app.modules.hr.interfaces.i_hr_service import IHRService
from app.modules.hr.repositories.hr_repository import HRRepository
from app.modules.hr.services.hr_service import HRService
from app.shared.dtos.api_response import APIResponse

router = APIRouter(prefix="/hr", tags=["Department: Human Resources"])


def get_hr_service(session: AsyncSession = Depends(get_db_session)) -> IHRService:
    repo = HRRepository(session)
    return HRService(repo)


@router.post(
    "/policies",
    response_model=APIResponse[HRPolicyResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new HR Policy (Restricted to HR Manager or System Admin)"
)
async def create_policy(
    request: Request,
    body: CreateHRPolicyRequest,
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER)),
    hr_service: IHRService = Depends(get_hr_service)
) -> APIResponse[HRPolicyResponse]:
    request_id = getattr(request.state, "request_id", None)
    
    # Non-admin managers can only manage policies within HR
    if current_user.role != UserRole.ADMIN and current_user.department.upper() != "HR":
        raise ForbiddenException("Only HR department managers or system admins can manage HR policies.")

    policy = await hr_service.create_policy(body)
    return APIResponse.success_response(
        data=policy,
        code=status.HTTP_201_CREATED,
        request_id=request_id
    )


@router.get(
    "/policies/{policy_code}",
    response_model=APIResponse[HRPolicyResponse],
    status_code=status.HTTP_200_OK,
    summary="Fetch an HR Policy by its unique code"
)
async def get_policy(
    request: Request,
    policy_code: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    hr_service: IHRService = Depends(get_hr_service)
) -> APIResponse[HRPolicyResponse]:
    request_id = getattr(request.state, "request_id", None)
    policy = await hr_service.get_by_code(policy_code)
    return APIResponse.success_response(
        data=policy,
        code=status.HTTP_200_OK,
        request_id=request_id
    )
