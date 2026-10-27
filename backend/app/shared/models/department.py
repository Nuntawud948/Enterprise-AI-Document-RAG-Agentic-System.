from typing import Optional
from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.models.base_entity import BaseAuditEntity


class Department(BaseAuditEntity):
    """
    Department master entity.
    Acts as the Single Source of Truth for department / tenant definitions
    and enforces referential integrity across users and documents.
    """
    __tablename__ = "departments"

    code: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
        comment="Unique department code, e.g. HR, IT, FINANCE, LEGAL, ENGINEERING"
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Human-readable department name"
    )
    description: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Detailed description or charter of the department"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether the department is active and can own resources"
    )
