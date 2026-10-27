from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.models.base_entity import DepartmentScopedEntity


class HRPolicy(DepartmentScopedEntity):
    """
    Department Slice: HR Policy Knowledge Entity.
    Demonstrates departmental domain isolation within Modular Monolith.
    """
    __tablename__ = "hr_policies"

    policy_code: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)  # Benefits, Leave, Code of Conduct
    summary: Mapped[str] = mapped_column(Text, nullable=False)
