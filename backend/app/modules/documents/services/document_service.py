import uuid
from typing import Optional, Union
from fastapi import BackgroundTasks
from app.core.exceptions import EntityNotFoundException, ForbiddenException, ValidationException
from app.core.security import AuthenticatedUser
from app.modules.auth.models.user import UserRole
from app.modules.documents.dtos.requests.document_requests import (
    DocumentCreateRequest,
    DocumentFilterRequest,
    VectorSearchRequest,
)
from app.modules.documents.dtos.responses.document_responses import (
    DocumentDetailResponse,
    VectorSearchResultItem,
    VectorSearchResponse,
)
from app.modules.documents.interfaces.i_chunk_repository import IChunkRepository
from app.modules.documents.interfaces.i_document_repository import IDocumentRepository
from app.modules.documents.interfaces.i_document_service import IDocumentService
from app.modules.documents.models.document import Document
from app.shared.dtos.pagination_dto import PagedResult, PaginationParams
from app.shared.tasks.ingestion_tasks import process_document_background_task


class DocumentService(IDocumentService):
    """
    Document management service business logic with multi-tenant department verification
    and Role-Based Access Control / Clearance Filtering.
    """

    def __init__(
        self,
        doc_repo: IDocumentRepository,
        chunk_repo: Optional[IChunkRepository] = None
    ):
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo

    async def create_document(
        self,
        request: DocumentCreateRequest,
        department: str,
        uploader_id: uuid.UUID,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> DocumentDetailResponse:
        doc = Document(
            title=request.title,
            file_name=request.file_name,
            file_type=request.file_type.lower(),
            file_size_bytes=request.file_size_bytes,
            status="QUEUED",
            classification_level=request.classification_level.upper() if request.classification_level else "INTERNAL",
            department=department.upper(),
            uploader_id=uploader_id
        )
        created = await self.doc_repo.create(doc)

        # Explicitly commit session before enqueuing background task
        # so row is visible to worker's independent AsyncSessionLocal() in PostgreSQL READ COMMITTED
        if hasattr(self.doc_repo, "session") and self.doc_repo.session:
            commit_fn = getattr(self.doc_repo.session, "commit", None)
            if callable(commit_fn):
                res = commit_fn()
                if hasattr(res, "__await__"):
                    await res

        if background_tasks is not None:
            background_tasks.add_task(
                process_document_background_task,
                document_id=created.id,
                department=department.upper(),
                content=getattr(request, "content", None)
            )

        return self._to_detail_response(created)

    async def get_document(
        self,
        document_id: uuid.UUID,
        user: Optional[Union[AuthenticatedUser, str]] = None,
        department: Optional[str] = None
    ) -> DocumentDetailResponse:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise EntityNotFoundException("Document", document_id)
        
        target = user if user is not None else department

        # Enforce Row-Level Security & RBAC:
        if isinstance(target, AuthenticatedUser):
            if not target.can_access_department(doc.department):
                raise ForbiddenException("Cross-department access violation. You cannot view this document.")
            # Check classification clearance
            if doc.classification_level not in target.get_allowed_clearances():
                raise ForbiddenException(
                    f"Insufficient security clearance to view '{doc.classification_level}' document."
                )
        elif target is not None:
            # Department string comparison
            if doc.department.upper() != str(target).upper():
                raise ForbiddenException("Cross-department access violation. You cannot view this document.")

        return self._to_detail_response(doc)

    async def list_documents(
        self,
        department: str,
        filter_req: DocumentFilterRequest,
        page_params: PaginationParams
    ) -> PagedResult[DocumentDetailResponse]:
        skip = (page_params.page - 1) * page_params.page_size
        docs = await self.doc_repo.list_by_department(
            department=department.upper(),
            status=filter_req.status,
            skip=skip,
            limit=page_params.page_size
        )
        total = await self.doc_repo.count_by_department(
            department=department.upper(),
            status=filter_req.status
        )
        items = [self._to_detail_response(d) for d in docs]
        return PagedResult(
            items=items,
            total=total,
            page=page_params.page,
            page_size=page_params.page_size
        )

    async def search_vectors(
        self,
        request: VectorSearchRequest,
        user: AuthenticatedUser
    ) -> VectorSearchResponse:
        if not self.chunk_repo:
            raise ValidationException("Vector retrieval repository is not configured.")

        # Data Leak Prevention:
        # Enforce user's department and user's allowed classification clearances
        allowed_clearances = user.get_allowed_clearances()
        results = await self.chunk_repo.search_similar_chunks(
            embedding=request.query_vector,
            department=user.department.upper(),
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            allowed_clearances=allowed_clearances
        )

        items = [
            VectorSearchResultItem(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                department=chunk.department,
                similarity_score=round(score, 4)
            )
            for chunk, score in results
        ]

        return VectorSearchResponse(
            query_department=user.department,
            clearance_level=user.clearance.value,
            total_results=len(items),
            results=items
        )

    async def delete_document(
        self,
        document_id: uuid.UUID,
        user: AuthenticatedUser
    ) -> bool:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise EntityNotFoundException("Document", document_id)

        # RBAC: Only ADMIN or MANAGER of that department can delete
        if user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
            raise ForbiddenException("Only Managers and Admins are authorized to delete documents.")

        if not user.can_access_department(doc.department):
            raise ForbiddenException("Cross-department delete violation.")

        return await self.doc_repo.delete(document_id)

    def _to_detail_response(self, doc: Document) -> DocumentDetailResponse:
        return DocumentDetailResponse(
            id=doc.id,
            title=doc.title,
            file_name=doc.file_name,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            status=doc.status,
            department=doc.department,
            classification_level=getattr(doc, "classification_level", None) or "INTERNAL",
            uploader_id=doc.uploader_id,
            chunks_count=0,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
