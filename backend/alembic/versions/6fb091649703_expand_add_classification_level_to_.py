"""expand_add_classification_level_to_documents

Revision ID: 6fb091649703
Revises: ef19bb04737f
Create Date: 2026-10-27 12:22:54.000000

Zero-Downtime Migration: Expand Phase
Adds 'classification_level' as a nullable column with server default 'INTERNAL'
so that live running instances do not fail during deployment.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '6fb091649703'
down_revision: Union[str, Sequence[str], None] = 'ef19bb04737f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Expand Phase:
    1. Add 'classification_level' as nullable with server_default='INTERNAL'.
    2. Add index for metadata filtering (TICKET-105 readiness).
    Existing production services can continue reading and writing without downtime.
    """
    op.add_column(
        'documents',
        sa.Column(
            'classification_level',
            sa.String(length=50),
            nullable=True,
            server_default='INTERNAL',
            comment='Security classification level: PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED'
        )
    )
    op.create_index(
        'ix_documents_classification_level',
        'documents',
        ['classification_level']
    )


def downgrade() -> None:
    """
    Rollback Expand Phase:
    Drop index and column safely.
    """
    op.drop_index('ix_documents_classification_level', table_name='documents')
    op.drop_column('documents', 'classification_level')
