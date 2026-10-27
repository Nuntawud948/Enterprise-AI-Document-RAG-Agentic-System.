from app.core.exceptions import ConflictException, UnauthorizedException
from app.core.security import create_access_token, get_password_hash, verify_password
from app.modules.auth.dtos.requests.auth_requests import LoginRequest, RegisterRequest
from app.modules.auth.dtos.responses.auth_responses import TokenResponse, UserProfileResponse
from app.modules.auth.interfaces.i_auth_service import IAuthService
from app.modules.auth.interfaces.i_user_repository import IUserRepository
from app.modules.auth.models.user import User, UserRole


class AuthService(IAuthService):
    """
    Core implementation of Authentication Business Logic.
    Decoupled from HTTP frameworks and easily unit-testable via mocked IUserRepository.
    """

    def __init__(self, user_repo: IUserRepository):
        self.user_repo = user_repo

    async def register(self, request: RegisterRequest) -> UserProfileResponse:
        existing = await self.user_repo.get_by_email(request.email)
        if existing:
            raise ConflictException(f"User with email '{request.email}' already exists.")

        user = User(
            email=request.email,
            hashed_password=get_password_hash(request.password),
            full_name=request.full_name,
            department=request.department.upper(),
            role=UserRole.MEMBER,
            is_active=True
        )
        created_user = await self.user_repo.create(user)
        return UserProfileResponse.model_validate(created_user)

    async def login(self, request: LoginRequest) -> TokenResponse:
        user = await self.user_repo.get_by_email(request.email)
        if not user or not verify_password(request.password, user.hashed_password):
            raise UnauthorizedException("Invalid email or password.")

        if not user.is_active:
            raise UnauthorizedException("User account is inactive. Please contact administrator.")

        token = create_access_token(
            subject=str(user.id),
            department=user.department,
            role=user.role.value
        )
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=UserProfileResponse.model_validate(user)
        )
