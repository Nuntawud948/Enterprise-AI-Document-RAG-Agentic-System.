from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.modules.auth.dtos.requests.auth_requests import LoginRequest, RegisterRequest
from app.modules.auth.dtos.responses.auth_responses import TokenResponse, UserProfileResponse
from app.modules.auth.interfaces.i_auth_service import IAuthService
from app.modules.auth.repositories.user_repository import UserRepository
from app.modules.auth.services.auth_service import AuthService
from app.shared.dtos.api_response import APIResponse

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])


def get_auth_service(session: AsyncSession = Depends(get_db_session)) -> IAuthService:
    """Dependency Provider for AuthService with injected repository."""
    user_repo = UserRepository(session)
    return AuthService(user_repo)


@router.post(
    "/register",
    response_model=APIResponse[UserProfileResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new employee into their department"
)
async def register(
    request: Request,
    body: RegisterRequest,
    auth_service: IAuthService = Depends(get_auth_service)
) -> APIResponse[UserProfileResponse]:
    request_id = getattr(request.state, "request_id", None)
    created_user = await auth_service.register(body)
    return APIResponse.success_response(
        data=created_user,
        code=status.HTTP_201_CREATED,
        request_id=request_id
    )


@router.post(
    "/login",
    response_model=APIResponse[TokenResponse],
    status_code=status.HTTP_200_OK,
    summary="User login returning signed JWT access token"
)
async def login(
    request: Request,
    body: LoginRequest,
    auth_service: IAuthService = Depends(get_auth_service)
) -> APIResponse[TokenResponse]:
    request_id = getattr(request.state, "request_id", None)
    token_response = await auth_service.login(body)
    return APIResponse.success_response(
        data=token_response,
        code=status.HTTP_200_OK,
        request_id=request_id
    )
