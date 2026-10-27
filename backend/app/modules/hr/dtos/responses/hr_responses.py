import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class HRPolicyResponse(BaseModel):
    """Response DTO for HR policy details."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    policy_code: str
    title: str
    category: str
    summary: str
    department: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
