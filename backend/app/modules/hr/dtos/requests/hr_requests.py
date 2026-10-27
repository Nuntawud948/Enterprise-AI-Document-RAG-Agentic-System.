from pydantic import BaseModel, Field


class CreateHRPolicyRequest(BaseModel):
    """Request DTO to register an HR Policy."""
    policy_code: str = Field(..., min_length=3, max_length=50, description="e.g. HR-LEAVE-2026")
    title: str = Field(..., min_length=3, max_length=255, description="Policy title")
    category: str = Field(..., min_length=2, max_length=100, description="e.g. Leave, Health, Compensation")
    summary: str = Field(..., min_length=10, description="Policy summary content")
