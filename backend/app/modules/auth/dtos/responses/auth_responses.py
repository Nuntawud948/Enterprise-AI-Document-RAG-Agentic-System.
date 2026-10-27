import uuid
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserProfileResponse(BaseModel):
    """
    Client-safe response DTO masking all sensitive credentials.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str
    department: str
    role: str
    is_active: bool


class TokenResponse(BaseModel):
    """Response DTO returning signed JWT bearer token and user profile."""
    access_token: str = Field(..., description="Signed JWT Bearer access token")
    token_type: str = Field(default="bearer", description="Token scheme")
    user: UserProfileResponse
