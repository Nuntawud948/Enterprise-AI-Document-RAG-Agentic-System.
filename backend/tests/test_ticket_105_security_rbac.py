import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.core.security import (
    AuthenticatedUser,
    ClassificationLevel,
    CLEARANCE_HIERARCHY,
    create_access_token,
    decode_access_token,
    get_current_user,
    require_roles,
)
from app.main import app
from app.modules.auth.models.user import UserRole
from app.modules.documents.dtos.requests.document_requests import VectorSearchRequest
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.modules.documents.services.document_service import DocumentService


client = TestClient(app)


def test_jwt_token_creation_and_claims_decoding():
    """Verify that JWT embeds subject, department, role, and clearance securely."""
    user_id = uuid.uuid4()
    token = create_access_token(
        subject=user_id,
        department="ENGINEERING",
        role=UserRole.MANAGER,
        clearance=ClassificationLevel.CONFIDENTIAL
    )
    assert token is not None

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == str(user_id)
    assert payload["department"] == "ENGINEERING"
    assert payload["role"] == "MANAGER"
    assert payload["clearance"] == "CONFIDENTIAL"
    assert "exp" in payload
    assert "iat" in payload


def test_clearance_hierarchy_and_permissions():
    """Verify classification clearance ladder: PUBLIC <= INTERNAL <= CONFIDENTIAL <= RESTRICTED."""
    member_user = AuthenticatedUser(
        id=uuid.uuid4(),
        email="dev@company.com",
        department="ENGINEERING",
        role=UserRole.MEMBER,
        clearance=ClassificationLevel.INTERNAL
    )
    manager_user = AuthenticatedUser(
        id=uuid.uuid4(),
        email="mgr@company.com",
        department="ENGINEERING",
        role=UserRole.MANAGER,
        clearance=ClassificationLevel.CONFIDENTIAL
    )
    admin_user = AuthenticatedUser(
        id=uuid.uuid4(),
        email="admin@company.com",
        department="IT",
        role=UserRole.ADMIN,
        clearance=ClassificationLevel.RESTRICTED
    )

    # Member can only access PUBLIC and INTERNAL
    assert member_user.get_allowed_clearances() == ["PUBLIC", "INTERNAL"]
    # Manager can access up to CONFIDENTIAL
    assert manager_user.get_allowed_clearances() == ["PUBLIC", "INTERNAL", "CONFIDENTIAL"]
    # Admin can access all clearances
    assert admin_user.get_allowed_clearances() == ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]

    # Department Isolation check
    assert member_user.can_access_department("ENGINEERING") is True
    assert member_user.can_access_department("LEGAL") is False
    # Admin has cross-department clearance
    assert admin_user.can_access_department("LEGAL") is True
    assert admin_user.can_access_department("HR") is True


@pytest.mark.asyncio
async def test_rbac_require_roles_guard():
    """Verify that require_roles allows matching roles and rejects unauthorized roles."""
    guard = require_roles(UserRole.ADMIN, UserRole.MANAGER)

    manager = AuthenticatedUser(
        id=uuid.uuid4(),
        email="mgr@company.com",
        department="HR",
        role=UserRole.MANAGER
    )
    member = AuthenticatedUser(
        id=uuid.uuid4(),
        email="dev@company.com",
        department="HR",
        role=UserRole.MEMBER
    )

    # Manager allowed
    res = await guard(current_user=manager)
    assert res == manager

    # Member forbidden
    with pytest.raises(ForbiddenException) as exc_info:
        await guard(current_user=member)
    assert "is not authorized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_data_leak_prevention_vector_search_sql_filtering():
    """
    [CRITICAL DATA LEAK PREVENTION]
    Verify that ChunkRepository.search_similar_chunks strictly enforces:
    1. Department isolation
    2. Document.classification_level IN (user_clearances)
    in the SQL query before computing similarity, preventing unauthorized vectors from leaking.
    """
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute.return_value = mock_result

    repo = ChunkRepository(session=mock_session)
    query_vector = [0.1] * 768
    allowed_clearances = ["PUBLIC", "INTERNAL"]

    await repo.search_similar_chunks(
        embedding=query_vector,
        department="ENGINEERING",
        top_k=5,
        score_threshold=0.7,
        allowed_clearances=allowed_clearances
    )

    call_args = mock_session.execute.call_args[0]
    stmt = call_args[0]
    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))

    # Verify JOIN with Document table for classification level check
    assert "JOIN documents" in compiled_sql
    # Verify strict department isolation
    assert "document_chunks.department =" in compiled_sql
    # Verify classification clearance filtering (Data Leak Prevention)
    assert "documents.classification_level IN" in compiled_sql
    # Verify cosine distance operator
    assert "<=>" in compiled_sql


@pytest.mark.asyncio
async def test_document_service_vector_search_clearance_filtering():
    """Verify DocumentService.search_vectors wires user clearance and department directly into repository."""
    mock_doc_repo = AsyncMock()
    mock_chunk_repo = AsyncMock()
    
    dummy_chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        department="FINANCE",
        chunk_index=0,
        content="Q3 Financial Performance Report",
        token_count=15
    )
    mock_chunk_repo.search_similar_chunks.return_value = [(dummy_chunk, 0.95)]

    service = DocumentService(doc_repo=mock_doc_repo, chunk_repo=mock_chunk_repo)

    user = AuthenticatedUser(
        id=uuid.uuid4(),
        email="accountant@company.com",
        department="FINANCE",
        role=UserRole.MEMBER,
        clearance=ClassificationLevel.INTERNAL
    )

    req = VectorSearchRequest(
        query_vector=[0.05] * 768,
        top_k=3,
        score_threshold=0.8
    )

    response = await service.search_vectors(req, user)

    assert response.query_department == "FINANCE"
    assert response.clearance_level == "INTERNAL"
    assert response.total_results == 1
    assert response.results[0].content == "Q3 Financial Performance Report"
    assert response.results[0].similarity_score == 0.95

    # Verify repository was invoked with user's exact department and clearances
    mock_chunk_repo.search_similar_chunks.assert_awaited_once_with(
        embedding=req.query_vector,
        department="FINANCE",
        top_k=3,
        score_threshold=0.8,
        allowed_clearances=["PUBLIC", "INTERNAL"]
    )


def test_endpoint_unauthorized_when_token_missing():
    """Verify that calling protected endpoint without Bearer token returns 401 standardized envelope."""
    doc_id = uuid.uuid4()
    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 401

    body = response.json()
    assert body["code"] == 401
    assert body["status"] == "error"
    assert "Authentication token was not provided" in body["errors"][0]


def test_endpoint_cross_department_forbidden_with_token():
    """Verify that user from HR cannot view document from LEGAL (returns 403 standardized envelope)."""
    from app.modules.documents.controllers.documents_controller import get_document_service
    from app.modules.documents.interfaces.i_document_service import IDocumentService

    mock_doc_service = AsyncMock(spec=IDocumentService)
    mock_doc_service.get_document.side_effect = ForbiddenException(
        "Cross-department access violation. You cannot view this document."
    )
    app.dependency_overrides[get_document_service] = lambda: mock_doc_service

    try:
        hr_user_id = uuid.uuid4()
        token = create_access_token(
            subject=hr_user_id,
            department="HR",
            role=UserRole.MEMBER,
            clearance=ClassificationLevel.INTERNAL
        )

        doc_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403
        body = response.json()
        assert body["code"] == 403
        assert body["status"] == "error"
        assert "Cross-department access violation" in body["errors"][0]
    finally:
        app.dependency_overrides.pop(get_document_service, None)
