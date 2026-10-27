from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request DTO for new user registration."""
    email: EmailStr = Field(..., description="Corporate email address")
    password: str = Field(..., min_length=8, description="Password with minimum 8 characters")
    full_name: str = Field(..., min_length=2, max_length=100, description="User full name")
    department: str = Field(..., min_length=2, max_length=50, description="Department name (e.g. HR, IT, LEGAL)")


class LoginRequest(BaseModel):
    """Request DTO for authentication."""
    email: EmailStr = Field(..., description="Corporate email address")
    password: str = Field(..., description="User password")
