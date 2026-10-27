import uuid
import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import EntityNotFoundException, ForbiddenException
from app.modules.documents.dtos.requests.document_requests import DocumentCreateRequest, DocumentFilterRequest
from app.modules.documents.interfaces.i_document_repository import IDocumentRepository
from app.modules.documents.models.document import Document
from app.modules.documents.services.document_service import DocumentService
from app.shared.dtos.pagination_dto import PaginationParams


@pytest.mark.asyncio
async def test_document_service_create_success():
    mock_repo = AsyncMock(spec=IDocumentRepository)

    async def mock_create(doc: Document):
        doc.id = uuid.uuid4()
        return doc

    mock_repo.create.side_effect = mock_create

    service = DocumentService(doc_repo=mock_repo)
    req = DocumentCreateRequest(
        title="Architecture Guide",
        file_name="arch.pdf",
        file_type="pdf",
        file_size_bytes=1024
    )
    uploader_id = uuid.uuid4()
    result = await service.create_document(
        request=req,
        department="ENGINEERING",
        uploader_id=uploader_id
    )

    assert result.title == "Architecture Guide"
    assert result.department == "ENGINEERING"
    assert result.status == "QUEUED"
    mock_repo.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_document_service_cross_department_forbidden():
    mock_repo = AsyncMock(spec=IDocumentRepository)
    doc_id = uuid.uuid4()
    # Document belongs to LEGAL
    doc = Document(
        id=doc_id,
        title="Legal Contract",
        file_name="contract.pdf",
        file_type="pdf",
        file_size_bytes=2048,
        status="PROCESSED",
        department="LEGAL",
        uploader_id=uuid.uuid4()
    )
    mock_repo.get_by_id.return_value = doc

    service = DocumentService(doc_repo=mock_repo)

    # ENGINEERING tries to access LEGAL doc
    with pytest.raises(ForbiddenException) as exc_info:
        await service.get_document(doc_id, department="ENGINEERING")

    assert "Cross-department access violation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_document_service_not_found():
    mock_repo = AsyncMock(spec=IDocumentRepository)
    mock_repo.get_by_id.return_value = None

    service = DocumentService(doc_repo=mock_repo)
    missing_id = uuid.uuid4()

    with pytest.raises(EntityNotFoundException):
        await service.get_document(missing_id, department="ENGINEERING")
