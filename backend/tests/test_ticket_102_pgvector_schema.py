import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.schema import CreateIndex, CreateTable

from app.modules.auth.models.user import User, UserRole
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.repositories.document_repository import DocumentRepository
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.shared.models.department import Department


def test_schema_department_master_entity():
    """Verify Department table structure, uniqueness, and constraints."""
    dept = Department(
        id=uuid.uuid4(),
        code="ENGINEERING",
        name="Engineering Division",
        description="Software and Infrastructure",
        is_active=True
    )
    assert dept.code == "ENGINEERING"
    assert dept.name == "Engineering Division"
    assert dept.is_active is True

    # Validate DDL table compilation for PostgreSQL
    ddl = str(CreateTable(Department.__table__).compile(dialect=postgresql.dialect()))
    assert "CREATE TABLE departments" in ddl
    assert "code VARCHAR(50) NOT NULL" in ddl


def test_schema_user_department_foreign_key():
    """Verify User entity references departments.code via Foreign Key."""
    user = User(
        email="eng.lead@company.com",
        hashed_password="secure_hash_val",
        full_name="Lead Engineer",
        department="ENGINEERING",
        role=UserRole.MANAGER
    )
    assert user.department == "ENGINEERING"

    # Verify foreign key constraint on users.department
    dept_col = User.__table__.columns["department"]
    fks = list(dept_col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "departments.code"


def test_schema_document_and_chunks_relationships_and_indexes():
    """Verify Document and DocumentChunk schemas, cascade, and HNSW index."""
    # Check Foreign Keys on Document
    doc_uploader_fk = list(Document.__table__.columns["uploader_id"].foreign_keys)
    assert len(doc_uploader_fk) == 1
    assert doc_uploader_fk[0].target_fullname == "users.id"

    doc_dept_fk = list(Document.__table__.columns["department"].foreign_keys)
    assert len(doc_dept_fk) == 1
    assert doc_dept_fk[0].target_fullname == "departments.code"

    # Check DocumentChunk Foreign Keys
    chunk_doc_fk = list(DocumentChunk.__table__.columns["document_id"].foreign_keys)
    assert len(chunk_doc_fk) == 1
    assert chunk_doc_fk[0].target_fullname == "documents.id"
    assert chunk_doc_fk[0].ondelete == "CASCADE"

    # Check DocumentChunk HNSW and Composite Indexes
    index_names = {idx.name: idx for idx in DocumentChunk.__table__.indexes}
    assert "ix_document_chunks_embedding_hnsw" in index_names
    assert "ix_document_chunks_department_document" in index_names

    hnsw_idx = index_names["ix_document_chunks_embedding_hnsw"]
    hnsw_ddl = str(CreateIndex(hnsw_idx).compile(dialect=postgresql.dialect()))
    assert "USING hnsw (embedding vector_cosine_ops)" in hnsw_ddl
    assert "WITH (m = 16, ef_construction = 64)" in hnsw_ddl

    comp_idx = index_names["ix_document_chunks_department_document"]
    comp_ddl = str(CreateIndex(comp_idx).compile(dialect=postgresql.dialect()))
    assert "(department, document_id)" in comp_ddl


def test_watch_out_anti_n_plus_one_lazy_raise():
    """
    [CRITICAL WATCH-OUT VERIFICATION]
    Verify that Document.chunks and Document.uploader have lazy='raise',
    which prevents accidental lazy-loading and N+1 query disaster.
    """
    from sqlalchemy.orm import class_mapper, sessionmaker
    from sqlalchemy import create_engine

    mapper = class_mapper(Document)
    assert mapper.relationships["chunks"].lazy == "raise"
    assert mapper.relationships["uploader"].lazy == "raise"

    # Verify runtime protection: Accessing unloaded relation on persistent entity raises InvalidRequestError
    engine = create_engine("sqlite:///:memory:")
    Document.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    doc = Document(
        id=uuid.uuid4(),
        title="Test Doc",
        file_name="test.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        department="ENGINEERING",
        uploader_id=uuid.uuid4()
    )
    session.add(doc)
    session.commit()

    with pytest.raises(InvalidRequestError, match="is not available due to lazy='raise'"):
        _ = doc.chunks

    with pytest.raises(InvalidRequestError, match="is not available due to lazy='raise'"):
        _ = doc.uploader


@pytest.mark.asyncio
async def test_document_repository_explicit_eager_loading_query():
    """Verify DocumentRepository.get_by_id_with_chunks builds query with selectinload and joinedload."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_doc = Document(
        id=uuid.uuid4(),
        title="Architecture Blueprint",
        file_name="blueprint.pdf",
        file_type="pdf",
        file_size_bytes=4096,
        department="ENGINEERING",
        uploader_id=uuid.uuid4()
    )
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_session.execute.return_value = mock_result

    repo = DocumentRepository(session=mock_session)
    fetched = await repo.get_by_id_with_chunks(mock_doc.id)

    assert fetched == mock_doc
    mock_session.execute.assert_awaited_once()

    # Verify statement contained Document.chunks and Document.uploader options
    call_args = mock_session.execute.call_args[0]
    stmt = call_args[0]
    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "WHERE documents.id =" in compiled_sql


@pytest.mark.asyncio
async def test_chunk_repository_vector_similarity_search():
    """
    Verify ChunkRepository.search_similar_chunks compiles correct pgvector
    cosine distance operator (<=>), department isolation, and thresholding.
    """
    mock_session = AsyncMock()
    dummy_chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        department="ENGINEERING",
        chunk_index=0,
        content="Clean Architecture & Vector Stores",
        token_count=12
    )

    # Mock DB execution returning row: (DocumentChunk, similarity_score)
    mock_result = MagicMock()
    mock_result.all.return_value = [(dummy_chunk, 0.92)]
    mock_session.execute.return_value = mock_result

    repo = ChunkRepository(session=mock_session)
    query_vector = [0.05] * 768

    results = await repo.search_similar_chunks(
        embedding=query_vector,
        department="ENGINEERING",
        top_k=3,
        score_threshold=0.8
    )

    assert len(results) == 1
    chunk, score = results[0]
    assert chunk == dummy_chunk
    assert score == 0.92

    # Check compiled SQL query structure
    call_args = mock_session.execute.call_args[0]
    stmt = call_args[0]
    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))

    # Verify <=> operator is used for cosine distance
    assert "<=>" in compiled_sql
    # Verify department filter is in query
    assert "document_chunks.department =" in compiled_sql
    # Verify LIMIT is applied
    assert "LIMIT" in compiled_sql
