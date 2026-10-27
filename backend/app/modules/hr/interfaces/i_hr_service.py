from abc import ABC, abstractmethod
from typing import List, Optional
from app.modules.hr.dtos.requests.hr_requests import CreateHRPolicyRequest
from app.modules.hr.dtos.responses.hr_responses import HRPolicyResponse


class IHRService(ABC):
    """Contract for HR Department business operations."""

    @abstractmethod
    async def create_policy(self, request: CreateHRPolicyRequest) -> HRPolicyResponse:
        """Create a new HR policy record."""
        pass

    @abstractmethod
    async def get_by_code(self, policy_code: str) -> HRPolicyResponse:
        """Retrieve policy by code."""
        pass

    @abstractmethod
    async def list_policies(self, category: Optional[str] = None) -> List[HRPolicyResponse]:
        """List all HR policies."""
        pass
