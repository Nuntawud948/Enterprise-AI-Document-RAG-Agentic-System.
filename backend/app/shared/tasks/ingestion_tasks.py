import asyncio
import logging
import os
import uuid
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import AsyncSessionLocal
from app.modules.documents.models.document import Document, DocumentChunk
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder

logger = logging.getLogger(__name__)


async def process_document_background_task(
    document_id: uuid.UUID,
    department: Optional[str] = None,
    raw_content: Optional[str] = None,
    content: Optional[str] = None,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    chunker: Optional[TextChunker] = None,
    embedder: Optional[VectorEmbedder] = None,
    **kwargs: Any,
) -> None:
    """
    Background worker task to parse, chunk, embed, and persist documents asynchronously.
    Prevents HTTP Request timeout and achieves sub-100ms endpoint response SLA (TICKET-304).

    Guarantees:
    - Runs in an isolated AsyncSessionLocal session independent of the HTTP request session.
    - Transitions document lifecycle: QUEUED -> PROCESSING -> SUCCESS (or FAILED on error).
    - Format-aware chunking (Markdown vs standard text) with null byte stripping.
    - Concurrent batch vector generation via VectorEmbedder.
    - Bulk persistence with strict multi-tenancy department isolation (chunk.department == doc.department).
    - Catches all exceptions, rolls back uncommitted chunks, and marks document FAILED without leaking exceptions.
    """
    logger.info(f"Starting background ingestion for document_id={document_id}, department={department}")
    factory = session_factory or AsyncSessionLocal

    try:
        async with factory() as session:
            # Step 1: Fetch document with retry polling to handle commit latency
            doc: Optional[Document] = None
            for attempt in range(3):
                doc = await session.get(Document, document_id)
                if doc is not None:
                    break
                if attempt < 2:
                    await asyncio.sleep(0.05)

            if not doc:
                logger.warning(f"Document {document_id} not found after retries; skipping background ingestion.")
                return

            doc.status = "PROCESSING"
            try:
                await session.commit()
            except Exception as proc_err:
                logger.error(f"Failed to transition document {document_id} to PROCESSING: {proc_err}")
                await session.rollback()
                return

            # Step 2: Extract and sanitize text content
            text_to_process = content if content is not None else raw_content
            if text_to_process is None and hasattr(doc, "file_path") and doc.file_path:
                try:
                    if os.path.exists(doc.file_path):
                        with open(doc.file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text_to_process = f.read()
                except Exception as read_err:
                    logger.warning(f"Failed to read file from path {doc.file_path}: {read_err}")

            if text_to_process is None:
                text_to_process = ""

            # Strip null bytes to prevent PostgreSQL DBAPI errors
            text_to_process = text_to_process.replace("\x00", "").strip()

            # Empty content handling: transition to SUCCESS with 0 chunks
            if not text_to_process:
                logger.info(f"Document {document_id} has empty content; marking SUCCESS with 0 chunks.")
                doc.status = "SUCCESS"
                await session.commit()
                return

            # Step 3: Instantiate RAG components if not injected
            active_chunker = chunker or TextChunker(chunk_size=500, chunk_overlap=50)
            active_embedder = embedder or VectorEmbedder(gemini_client=GeminiClient())

            doc_metadata = {
                "document_id": str(doc.id),
                "department": doc.department,
                "file_name": doc.file_name,
                "classification_level": getattr(doc, "classification_level", "INTERNAL"),
            }

            # File-type aware chunking (supports MIME 'text/markdown' and .md/.markdown extensions)
            file_type_str = (doc.file_type or "").lower()
            file_name_str = (doc.file_name or "").lower()
            is_markdown = (
                "markdown" in file_type_str
                or file_type_str == "md"
                or file_name_str.endswith((".md", ".markdown"))
            )

            if is_markdown:
                chunk_results = active_chunker.split_markdown(
                    text_to_process,
                    document_metadata=doc_metadata,
                )
            else:
                chunk_results = active_chunker.split_text_with_metadata(
                    text_to_process,
                    document_metadata=doc_metadata,
                )

            if not chunk_results:
                logger.info(f"Chunker produced 0 chunks for document {document_id}; setting status to SUCCESS.")
                doc.status = "SUCCESS"
                await session.commit()
                return

            # Step 4: Batch vector generation
            chunk_texts = [cr.text for cr in chunk_results]
            embeddings = await active_embedder.embed_batch(chunk_texts)

            # Step 5: Build DocumentChunk entities with STRICT department isolation
            chunks = []
            for i, (res, emb) in enumerate(zip(chunk_results, embeddings)):
                chunk = DocumentChunk(
                    document_id=doc.id,
                    department=doc.department,  # STRICT department isolation!
                    chunk_index=i,
                    content=res.text,
                    embedding=emb,
                    token_count=res.token_count,
                    chunk_metadata={
                        **res.metadata,
                        "start_char": res.start_char,
                        "end_char": res.end_char,
                    },
                )
                chunk.start_char = res.start_char
                chunk.end_char = res.end_char
                chunks.append(chunk)

            # Bulk persistence via ChunkRepository
            chunk_repo = ChunkRepository(session)
            await chunk_repo.create_chunks(chunks)

            # Transition document status to SUCCESS
            doc.status = "SUCCESS"
            await session.commit()
            logger.info(
                f"Successfully completed background ingestion for document {document_id}: "
                f"persisted {len(chunks)} chunks."
            )

    except Exception as exc:
        logger.exception(f"Background ingestion failed for document {document_id}: {exc}")
        # Rollback active session and attempt status transition to FAILED
        try:
            await session.rollback()
            doc_to_fail = await session.get(Document, document_id)
            if doc_to_fail:
                doc_to_fail.status = "FAILED"
                await session.commit()
                return
        except Exception as rollback_err:
            logger.warning(f"Could not update status on active session after rollback: {rollback_err}")

        # Fallback to fresh session if active session was invalidated or closed
        try:
            async with factory() as fresh_session:
                doc_to_fail = await fresh_session.get(Document, document_id)
                if doc_to_fail:
                    doc_to_fail.status = "FAILED"
                    await fresh_session.commit()
        except Exception as fresh_err:
            logger.error(f"Failed to record FAILED status in fresh session for document {document_id}: {fresh_err}")
