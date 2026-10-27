import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BaseAuditEntity(Base):
    """
    Abstract Base Class for all Domain Entities.
    Includes UUID primary key and audit timestamps.
    """
    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class DepartmentScopedEntity(BaseAuditEntity):
    """
    Abstract Base Class for Multi-Tenant / Department-sliced entities.
    Enforces a 'department' column for row-level isolation.
    """
    __abstract__ = True

    department: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("departments.code", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Department code for multi-tenancy row-level isolation referencing departments.code"
    )
