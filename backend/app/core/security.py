import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union
import bcrypt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.modules.auth.models.user import UserRole


class ClassificationLevel(str, enum.Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


# Clearance hierarchy: each level includes all lower levels
CLEARANCE_HIERARCHY: Dict[str, List[str]] = {
    ClassificationLevel.PUBLIC.value: [ClassificationLevel.PUBLIC.value],
    ClassificationLevel.INTERNAL.value: [
        ClassificationLevel.PUBLIC.value,
        ClassificationLevel.INTERNAL.value,
    ],
    ClassificationLevel.CONFIDENTIAL.value: [
        ClassificationLevel.PUBLIC.value,
        ClassificationLevel.INTERNAL.value,
        ClassificationLevel.CONFIDENTIAL.value,
    ],
    ClassificationLevel.RESTRICTED.value: [
        ClassificationLevel.PUBLIC.value,
        ClassificationLevel.INTERNAL.value,
        ClassificationLevel.CONFIDENTIAL.value,
        ClassificationLevel.RESTRICTED.value,
    ],
}

# Default clearance mapped from User Role
ROLE_DEFAULT_CLEARANCE: Dict[UserRole, ClassificationLevel] = {
    UserRole.MEMBER: ClassificationLevel.INTERNAL,
    UserRole.MANAGER: ClassificationLevel.CONFIDENTIAL,
    UserRole.ADMIN: ClassificationLevel.RESTRICTED,
}


class AuthenticatedUser(BaseModel):
    """Domain representation of an authenticated user extracted from JWT."""
    id: uuid.UUID
    email: str
    department: str
    role: UserRole
    clearance: ClassificationLevel = ClassificationLevel.INTERNAL

    def can_access_department(self, target_department: str) -> bool:
        """Admins can access all departments; other users strictly their own."""
        if self.role == UserRole.ADMIN:
            return True
        return self.department.upper() == target_department.upper()

    def get_allowed_clearances(self) -> List[str]:
        """Returns the list of document classifications this user is permitted to retrieve."""
        return CLEARANCE_HIERARCHY.get(self.clearance.value, [ClassificationLevel.PUBLIC.value])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Generates bcrypt salt and hashes the plain password (max 72 bytes per bcrypt standard)."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def create_access_token(
    subject: Union[str, Any],
    department: str,
    role: Union[str, UserRole],
    clearance: Optional[Union[str, ClassificationLevel]] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Creates a signed JWT access token embedding user identity, department, role, and clearance.
    """
    role_str = role.value if isinstance(role, UserRole) else str(role)

    if clearance is None:
        try:
            user_role = UserRole(role_str)
            clearance_str = ROLE_DEFAULT_CLEARANCE.get(user_role, ClassificationLevel.INTERNAL).value
        except ValueError:
            clearance_str = ClassificationLevel.INTERNAL.value
    else:
        clearance_str = clearance.value if isinstance(clearance, ClassificationLevel) else str(clearance)

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(subject),
        "department": department,
        "role": role_str,
        "clearance": clearance_str,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decodes and validates JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except Exception:
        return None


# FastAPI Bearer Token extractor
http_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> AuthenticatedUser:
    """
    FastAPI Dependency: Authenticates request using Bearer JWT.
    Enforces standardized UnauthorizedException envelope upon any auth failure.
    """
    if not auth_credentials or not auth_credentials.credentials:
        raise UnauthorizedException("Authentication token was not provided in Authorization header.")

    token = auth_credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id_str: Optional[str] = payload.get("sub")
        department: Optional[str] = payload.get("department")
        role_str: Optional[str] = payload.get("role")
        clearance_str: Optional[str] = payload.get("clearance", ClassificationLevel.INTERNAL.value)

        if not user_id_str or not department or not role_str:
            raise UnauthorizedException("Invalid token claims: subject, department, or role missing.")

        user_id = uuid.UUID(user_id_str)
        role = UserRole(role_str)
        clearance = ClassificationLevel(clearance_str)

        return AuthenticatedUser(
            id=user_id,
            email=payload.get("email", f"{user_id}@company.internal"),
            department=department,
            role=role,
            clearance=clearance
        )
    except (JWTError, ValueError, KeyError):
        raise UnauthorizedException("Invalid or expired authentication token.")


def require_roles(*allowed_roles: UserRole) -> Callable:
    """
    FastAPI Dependency Factory: Enforces Role-Based Access Control (RBAC).
    Raises ForbiddenException if the authenticated user lacks the required role.
    """
    async def role_checker(
        current_user: AuthenticatedUser = Depends(get_current_user)
    ) -> AuthenticatedUser:
        if current_user.role not in allowed_roles:
            role_names = [r.value for r in allowed_roles]
            raise ForbiddenException(
                f"Role '{current_user.role.value}' is not authorized. Required roles: {role_names}"
            )
        return current_user

    return role_checker
