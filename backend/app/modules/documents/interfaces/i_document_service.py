import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from fastapi import BackgroundTasks
from app.core.security import AuthenticatedUser
from app.modules.documents.dtos.requests.document_requests import (
    DocumentCreateRequest,
    DocumentFilterRequest,
    VectorSearchRequest,
)
from app.modules.documents.dtos.responses.document_responses import (
    DocumentDetailResponse,
    VectorSearchResponse,
)
from app.shared.dtos.pagination_dto import PagedResult, PaginationParams


class IDocumentService(ABC):
    """Contract for Enterprise Document Ingestion, Retrieval & Vector Search."""

    @abstractmethod
    async def create_document(
        self,
        request: DocumentCreateRequest,
        department: str,
        uploader_id: uuid.UUID,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> DocumentDetailResponse:
        """Create and queue document for ingestion in a department."""
        pass

    @abstractmethod
    async def get_document(
        self,
        document_id: uuid.UUID,
        user: Optional[Union[AuthenticatedUser, str]] = None,
        department: Optional[str] = None
    ) -> DocumentDetailResponse:
        """Fetch document guaranteeing department row-level isolation & RBAC."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        department: str,
        filter_req: DocumentFilterRequest,
        page_params: PaginationParams
    ) -> PagedResult[DocumentDetailResponse]:
        """Fetch paginated documents for the caller's department."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        request: VectorSearchRequest,
        user: AuthenticatedUser
    ) -> VectorSearchResponse:
        """
        Perform vector similarity search enforcing department isolation
        and data leak prevention via classification clearances.
        """
        pass

    @abstractmethod
    async def delete_document(
        self,
        document_id: uuid.UUID,
        user: AuthenticatedUser
    ) -> bool:
        """Delete document strictly verifying department and role permissions."""
        pass
