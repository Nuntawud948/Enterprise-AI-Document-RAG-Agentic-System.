from abc import ABC, abstractmethod
from app.modules.auth.dtos.requests.auth_requests import LoginRequest, RegisterRequest
from app.modules.auth.dtos.responses.auth_responses import TokenResponse, UserProfileResponse


class IAuthService(ABC):
    """Business logic contract for authentication and user onboarding."""

    @abstractmethod
    async def register(self, request: RegisterRequest) -> UserProfileResponse:
        """Register a new user into a specific department."""
        pass

    @abstractmethod
    async def login(self, request: LoginRequest) -> TokenResponse:
        """Authenticate user credentials and issue signed JWT access token."""
        pass
