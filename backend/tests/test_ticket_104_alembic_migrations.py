import io
import os
from unittest.mock import patch
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic import command

from app.core.config import settings
from app.core.database import Base
from app.shared.models.department import Department
from app.modules.auth.models.user import User
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.hr.models.hr_policy import HRPolicy


def get_alembic_config() -> Config:
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ini_path = os.path.join(backend_dir, "alembic.ini")
    config = Config(ini_path)
    config.set_main_option("script_location", os.path.join(backend_dir, "alembic"))
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
    return config


def test_alembic_metadata_registered_entities():
    """Verify all domain models are registered into Base.metadata for Alembic autogeneration."""
    expected_tables = {
        "departments",
        "users",
        "documents",
        "document_chunks",
        "hr_policies"
    }
    registered_tables = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(registered_tables), f"Missing tables in metadata: {expected_tables - registered_tables}"


def test_alembic_revision_chain_integrity():
    """Verify that Alembic migrations have a valid continuous chain without gaps or broken branches."""
    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)

    # Verify head revision
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected exactly 1 head revision, got {heads}"
    head_rev = heads[0]
    assert head_rev == "6fb091649703"

    # Walk through revisions from head back to base
    rev_sequence = []
    curr = script.get_revision(head_rev)
    while curr is not None:
        rev_sequence.append(curr.revision)
        curr = script.get_revision(curr.down_revision)

    # Order from base to head: initial_schema -> expand_add_classification_level
    rev_sequence.reverse()
    assert rev_sequence == ["ef19bb04737f", "6fb091649703"]


def test_alembic_offline_sql_generation_expand_contract():
    """
    Verify offline SQL generation for Zero-Downtime Migration:
    Confirms 'CREATE EXTENSION vector', tables, HNSW index, and Expand column.
    """
    config = get_alembic_config()
    
    # Capture stdout buffer for --sql upgrade head
    buffer = io.StringIO()
    with patch("sys.stdout", buffer):
        command.upgrade(config, "head", sql=True)
    
    sql_content = buffer.getvalue()
    
    # Assert transactional wrapper
    assert "BEGIN;" in sql_content
    assert "COMMIT;" in sql_content
    
    # Assert pgvector extension creation
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in sql_content
    
    # Assert relational tables
    assert "CREATE TABLE departments" in sql_content
    assert "CREATE TABLE users" in sql_content
    assert "CREATE TABLE documents" in sql_content
    assert "CREATE TABLE document_chunks" in sql_content
    assert "CREATE TABLE hr_policies" in sql_content

    # Assert HNSW index
    assert "USING hnsw (embedding vector_cosine_ops)" in sql_content
    assert "WITH (m = 16, ef_construction = 64)" in sql_content

    # Assert Zero-Downtime Expand Migration (nullable with server_default)
    assert "ALTER TABLE documents ADD COLUMN classification_level VARCHAR(50) DEFAULT 'INTERNAL'" in sql_content
    assert "CREATE INDEX ix_documents_classification_level" in sql_content


def test_alembic_offline_downgrade_sql_generation():
    """Verify offline SQL generation for rollback (downgrade) cleanly drops created elements in reverse order."""
    config = get_alembic_config()

    buffer = io.StringIO()
    with patch("sys.stdout", buffer):
        command.downgrade(config, "6fb091649703:ef19bb04737f", sql=True)

    sql_content = buffer.getvalue()
    assert "DROP INDEX ix_documents_classification_level;" in sql_content
    assert "ALTER TABLE documents DROP COLUMN classification_level;" in sql_content
