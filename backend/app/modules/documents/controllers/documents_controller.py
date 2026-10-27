import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.security import AuthenticatedUser, get_current_user
from app.modules.documents.dtos.requests.document_requests import (
    DocumentCreateRequest,
    DocumentFilterRequest,
    VectorSearchRequest,
)
from app.modules.documents.dtos.responses.document_responses import (
    DocumentDetailResponse,
    VectorSearchResponse,
)
from app.modules.documents.interfaces.i_document_service import IDocumentService
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.modules.documents.repositories.document_repository import DocumentRepository
from app.modules.documents.services.document_service import DocumentService
from app.shared.dtos.api_response import APIResponse
from app.shared.dtos.pagination_dto import PagedResult, PaginationParams

router = APIRouter(prefix="/documents", tags=["Enterprise Document Management"])


def get_document_service(session: AsyncSession = Depends(get_db_session)) -> IDocumentService:
    """Dependency Provider for DocumentService with injected repositories."""
    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)
    return DocumentService(doc_repo=doc_repo, chunk_repo=chunk_repo)


@router.post(
    "",
    response_model=APIResponse[DocumentDetailResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new document metadata for ingestion (Scoped to authenticated user)"
)
async def create_document(
    request: Request,
    body: DocumentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user),
    doc_service: IDocumentService = Depends(get_document_service)
) -> APIResponse[DocumentDetailResponse]:
    request_id = getattr(request.state, "request_id", None)
    doc = await doc_service.create_document(
        request=body,
        department=current_user.department,
        uploader_id=current_user.id,
        background_tasks=background_tasks
    )
    return APIResponse.success_response(
        data=doc,
        code=status.HTTP_201_CREATED,
        request_id=request_id
    )


@router.get(
    "/{document_id}",
    response_model=APIResponse[DocumentDetailResponse],
    status_code=status.HTTP_200_OK,
    summary="Retrieve document details with department row-level isolation & RBAC"
)
async def get_document_by_id(
    request: Request,
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    doc_service: IDocumentService = Depends(get_document_service)
) -> APIResponse[DocumentDetailResponse]:
    request_id = getattr(request.state, "request_id", None)
    doc = await doc_service.get_document(document_id, current_user)
    return APIResponse.success_response(
        data=doc,
        code=status.HTTP_200_OK,
        request_id=request_id
    )


@router.get(
    "",
    response_model=APIResponse[PagedResult[DocumentDetailResponse]],
    status_code=status.HTTP_200_OK,
    summary="List department documents with pagination"
)
async def list_documents(
    request: Request,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: AuthenticatedUser = Depends(get_current_user),
    doc_service: IDocumentService = Depends(get_document_service)
) -> APIResponse[PagedResult[DocumentDetailResponse]]:
    request_id = getattr(request.state, "request_id", None)
    filter_req = DocumentFilterRequest(status=status_filter)
    page_params = PaginationParams(page=page, page_size=page_size)
    result = await doc_service.list_documents(
        department=current_user.department,
        filter_req=filter_req,
        page_params=page_params
    )
    return APIResponse.success_response(
        data=result,
        code=status.HTTP_200_OK,
        request_id=request_id
    )


@router.post(
    "/search",
    response_model=APIResponse[VectorSearchResponse],
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search with Data Leak Prevention (RBAC & Clearance Pre-filter)"
)
async def search_documents_vector(
    request: Request,
    body: VectorSearchRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    doc_service: IDocumentService = Depends(get_document_service)
) -> APIResponse[VectorSearchResponse]:
    request_id = getattr(request.state, "request_id", None)
    results = await doc_service.search_vectors(body, current_user)
    return APIResponse.success_response(
        data=results,
        code=status.HTTP_200_OK,
        request_id=request_id
    )


@router.delete(
    "/{document_id}",
    response_model=APIResponse[bool],
    status_code=status.HTTP_200_OK,
    summary="Delete document (Requires MANAGER or ADMIN role)"
)
async def delete_document(
    request: Request,
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    doc_service: IDocumentService = Depends(get_document_service)
) -> APIResponse[bool]:
    request_id = getattr(request.state, "request_id", None)
    deleted = await doc_service.delete_document(document_id, current_user)
    return APIResponse.success_response(
        data=deleted,
        code=status.HTTP_200_OK,
        request_id=request_id
    )
