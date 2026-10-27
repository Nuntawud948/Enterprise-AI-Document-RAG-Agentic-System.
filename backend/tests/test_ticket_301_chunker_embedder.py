import asyncio
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CircuitBreakerOpenError, VectorServiceError
from app.modules.documents.models.document import DocumentChunk
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import ChunkResult, TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder


# ============================================================================
# 1. TextChunker Constructor & Input Validation Tests
# ============================================================================

def test_chunker_valid_initialization():
    """Verify default initialization and custom parameter assignment."""
    chunker = TextChunker()
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50
    assert chunker.separators == ["\n\n", "\n", ". ", " ", ""]

    custom = TextChunker(chunk_size=200, chunk_overlap=20, separators=["\n", " "])
    assert custom.chunk_size == 200
    assert custom.chunk_overlap == 20
    assert custom.separators == ["\n", " "]


@pytest.mark.parametrize("invalid_size", [0, -1, -500])
def test_chunker_invalid_chunk_size_raises(invalid_size: int):
    """Verify chunk_size <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=invalid_size)


@pytest.mark.parametrize("invalid_overlap", [-1, -50])
def test_chunker_negative_overlap_raises(invalid_overlap: int):
    """Verify negative chunk_overlap raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=500, chunk_overlap=invalid_overlap)


@pytest.mark.parametrize(
    "size,overlap",
    [(100, 100), (100, 101), (50, 100)],
)
def test_chunker_overlap_greater_or_equal_to_size_raises(size: int, overlap: int):
    """Verify chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextChunker(chunk_size=size, chunk_overlap=overlap)


# ============================================================================
# 2. TextChunker Edge Cases & Basic Splitting Tests
# ============================================================================

def test_chunker_empty_and_whitespace_text():
    """Verify empty string and whitespace-only text return empty lists."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  \n   ") == []

    assert chunker.split_text_with_metadata("") == []
    assert chunker.split_text_with_metadata("   \n\t  ") == []

    assert chunker.split_markdown("") == []
    assert chunker.split_markdown("   \n\t  ") == []


def test_chunker_short_text_single_chunk():
    """Verify text shorter than or equal to chunk_size returns a single chunk."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Enterprise RAG document chunking engine."

    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text

    exact_text = "A" * 100
    chunks_exact = chunker.split_text(exact_text)
    assert len(chunks_exact) == 1
    assert chunks_exact[0] == exact_text


def test_chunker_massive_string_without_separators():
    """
    Adversarial test: A continuous string of 20,000 characters with no whitespace
    must be sliced cleanly by character fallback without recursion error or memory exhaustion.
    """
    chunk_size = 200
    chunk_overlap = 50
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    massive_text = "X" * 20000
    chunks = chunker.split_text(massive_text)

    step = chunk_size - chunk_overlap
    expected_count = (len(massive_text) + step - 1) // step
    assert len(chunks) == expected_count

    for chunk in chunks:
        assert len(chunk) <= chunk_size


def test_chunker_empty_separator_guard():
    """Verify that an explicit empty separator '' does not raise ValueError: empty separator."""
    chunker = TextChunker(chunk_size=10, chunk_overlap=2, separators=[""])
    text = "ABCDEFGHIJKLMNO"
    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 10


# ============================================================================
# 3. Recursive Splitting & Delimiter Preservation Tests
# ============================================================================

def test_chunker_paragraph_splitting_preserves_delimiter():
    """Verify paragraph splitting combines small paragraphs and preserves delimiters."""
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    text = "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three is here."

    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    # Check that delimiter is preserved in the joined chunks
    assert "\n\n" in chunks[0]


def test_chunker_sentence_fallback():
    """Verify fallback to sentence boundaries when paragraph exceeds chunk_size."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "First short sentence. Second sentence goes here. Third sentence is quite long indeed."

    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 50


def test_chunker_word_fallback():
    """Verify fallback to word boundary (' ') when sentences exceed chunk_size."""
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    text = "supercalifragilisticexpialidocious anotherword quitealongtoken stringofwords"

    chunks = chunker.split_text(text)
    assert len(chunks) >= 3


# ============================================================================
# 4. Metadata, Monotonic Offsets & ChunkResult Model Tests
# ============================================================================

def test_chunk_result_model_compatibility():
    """Verify ChunkResult works with both 'text' and 'content' accessors."""
    res1 = ChunkResult(
        chunk_index=0,
        text="Sample text",
        token_count=2,
        metadata={"key": "val"},
        start_char=0,
        end_char=11,
    )
    assert res1.text == "Sample text"
    assert res1.content == "Sample text"
    assert res1.token_count == 2
    assert res1.metadata["key"] == "val"

    # Backward-compatibility: initializing with content should populate text
    res2 = ChunkResult(
        chunk_index=1,
        content="Alternative content",
        token_count=4,
        start_char=15,
        end_char=34,
    )
    assert res2.text == "Alternative content"
    assert res2.content == "Alternative content"


def test_split_text_with_metadata_monotonic_offsets():
    """
    Verify character offsets (start_char, end_char) are strictly monotonic and accurate,
    even with overlap and repeating phrases.
    """
    chunker = TextChunker(chunk_size=40, chunk_overlap=10)
    # Repeating phrases can cause search cursor traps if search doesn't advance past prev_start + 1
    text = "Alpha beta gamma. Alpha beta gamma. Delta epsilon zeta. Delta epsilon zeta."

    results = chunker.split_text_with_metadata(text, metadata={"doc_id": "123"})
    assert len(results) >= 2

    prev_start = -1
    for i, res in enumerate(results):
        assert res.chunk_index == i
        assert res.metadata["doc_id"] == "123"
        assert res.start_char >= prev_start  # Monotonic progression
        assert res.start_char < res.end_char
        assert res.token_count == max(1, len(res.text) // 4)

        # Verify substring in source text matches chunk text at start_char
        source_slice = text[res.start_char : res.end_char]
        assert source_slice == res.text
        prev_start = res.start_char


# ============================================================================
# 5. Markdown Splitting Hierarchy & Code Block Fence Guard Tests
# ============================================================================

def test_split_markdown_headers_hierarchy():
    """Verify split_markdown captures h1, h2, h3 hierarchy in metadata."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    markdown = """# System Overview
This is the main overview of the enterprise platform.

## Architecture
Details on the modular monolith clean architecture.

### Database Layer
pgvector PostgreSQL schema with row level security.

## Deployment
Docker and Kubernetes deployment guide.
"""

    results = chunker.split_markdown(markdown)
    assert len(results) == 4

    assert results[0].metadata["headers"] == {"h1": "System Overview"}
    assert "main overview" in results[0].text

    assert results[1].metadata["headers"] == {"h1": "System Overview", "h2": "Architecture"}
    assert "modular monolith" in results[1].text

    assert results[2].metadata["headers"] == {
        "h1": "System Overview",
        "h2": "Architecture",
        "h3": "Database Layer",
    }
    assert "pgvector PostgreSQL" in results[2].text

    # Sibling H2 clears previous H2 and H3
    assert results[3].metadata["headers"] == {"h1": "System Overview", "h2": "Deployment"}
    assert "Kubernetes deployment" in results[3].text


def test_split_markdown_preamble_before_first_header():
    """Verify preamble text before the first markdown header is captured with empty headers."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    markdown = """Welcome to the documentation.
This is preamble text before any header.

# Getting Started
Here is how to get started.
"""

    results = chunker.split_markdown(markdown)
    assert len(results) == 2

    assert results[0].metadata["headers"] == {}
    assert "Welcome to the documentation" in results[0].text
    assert results[0].chunk_index == 0

    assert results[1].metadata["headers"] == {"h1": "Getting Started"}
    assert "Here is how to get started" in results[1].text
    assert results[1].chunk_index == 1


def test_split_markdown_code_fence_guard():
    """
    Verify lines starting with '#' inside code blocks (```) are NOT treated as headers.
    """
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    markdown = """# Python Guide

Here is some sample code:
```python
# This is a comment inside code block, NOT a header!
def add(a, b):
    # Another comment
    return a + b
```

## Next Topic
After the code block.
"""

    results = chunker.split_markdown(markdown)
    assert len(results) == 2

    assert results[0].metadata["headers"] == {"h1": "Python Guide"}
    assert "# This is a comment inside code block" in results[0].text

    assert results[1].metadata["headers"] == {"h1": "Python Guide", "h2": "Next Topic"}
    assert "After the code block" in results[1].text


def test_split_markdown_oversized_section_offset_accuracy():
    """
    Verify oversized sections are recursively split and their start_char and end_char
    are adjusted by section_offset so they point accurately to the root document.
    """
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)
    markdown = (
        "# Short Header\n"
        "Brief content.\n\n"
        "# Big Section\n"
        "This is an oversized section that contains significantly more text than eighty characters. "
        "It will be recursively split into multiple chunks, and every single chunk must have its "
        "start_char and end_char accurately offset relative to the very beginning of the document."
    )

    results = chunker.split_markdown(markdown)
    assert len(results) >= 3

    # Section 1
    assert results[0].metadata["headers"] == {"h1": "Short Header"}
    assert markdown[results[0].start_char : results[0].end_char] == results[0].text

    # Big section sub-chunks
    for chunk in results[1:]:
        assert chunk.metadata["headers"] == {"h1": "Big Section"}
        # Ponytail Section Offset Fix check: root slice matches chunk text
        root_slice = markdown[chunk.start_char : chunk.end_char]
        assert root_slice == chunk.text, f"Offset mismatch: expected '{chunk.text}', got '{root_slice}'"


def test_split_markdown_without_headers():
    """Verify markdown without any headers is chunked gracefully."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    markdown = "Just a plain markdown document without any headers.\n\nSecond paragraph here."

    results = chunker.split_markdown(markdown)
    assert len(results) >= 1
    for res in results:
        assert res.metadata["headers"] == {}


# ============================================================================
# 6. VectorEmbedder Tests
# ============================================================================

@pytest.mark.asyncio
async def test_vector_embedder_constructor_validation():
    """Verify VectorEmbedder validates batch_size > 0."""
    mock_client = AsyncMock(spec=GeminiClient)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        VectorEmbedder(gemini_client=mock_client, batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        VectorEmbedder(gemini_client=mock_client, batch_size=-5)

    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=16)
    assert embedder.batch_size == 16


@pytest.mark.asyncio
async def test_vector_embedder_embed_chunk():
    """Verify embed_chunk delegates directly to gemini_client.generate_embedding."""
    mock_client = AsyncMock(spec=GeminiClient)
    dummy_vector = [0.1] * 768
    mock_client.generate_embedding.return_value = dummy_vector

    embedder = VectorEmbedder(gemini_client=mock_client)
    res = await embedder.embed_chunk("Hello world")

    assert res == dummy_vector
    mock_client.generate_embedding.assert_awaited_once_with("Hello world")


@pytest.mark.asyncio
async def test_vector_embedder_embed_batch_empty():
    """Verify embed_batch([]) returns [] immediately without calling API."""
    mock_client = AsyncMock(spec=GeminiClient)
    embedder = VectorEmbedder(gemini_client=mock_client)

    res = await embedder.embed_batch([])
    assert res == []
    mock_client.generate_embedding.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_embedder_embed_batch_slicing_and_concurrency():
    """Verify embed_batch correctly slices chunks into batches and processes concurrently."""
    mock_client = AsyncMock(spec=GeminiClient)
    dummy_vector = [0.05] * 768
    mock_client.generate_embedding.return_value = dummy_vector

    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=2)
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]

    results = await embedder.embed_batch(chunks)

    assert len(results) == 5
    assert all(vec == dummy_vector for vec in results)
    assert mock_client.generate_embedding.await_count == 5


@pytest.mark.asyncio
async def test_vector_embedder_embed_batch_invalid_batch_size_arg():
    """Verify embed_batch raises ValueError when passed invalid batch_size."""
    mock_client = AsyncMock(spec=GeminiClient)
    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=16)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        await embedder.embed_batch(["text"], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        await embedder.embed_batch(["text"], batch_size=-2)


@pytest.mark.asyncio
async def test_vector_embedder_propagates_exceptions():
    """Verify exceptions from GeminiClient (e.g. CircuitBreakerOpenError) propagate cleanly."""
    mock_client = AsyncMock(spec=GeminiClient)
    mock_client.generate_embedding.side_effect = CircuitBreakerOpenError(
        service_name="GeminiClient", recovery_time_remaining=25.0
    )

    embedder = VectorEmbedder(gemini_client=mock_client)
    with pytest.raises(CircuitBreakerOpenError):
        await embedder.embed_chunk("Failing text")


# ============================================================================
# 7. ChunkRepository create_chunks Tests
# ============================================================================

@pytest.mark.asyncio
async def test_chunk_repository_create_chunks_success():
    """Verify create_chunks persists chunks in bulk via add_all and flush."""
    mock_session = AsyncMock(spec=AsyncSession)
    repo = ChunkRepository(session=mock_session)

    doc_id = uuid.uuid4()
    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=0,
            content="Chunk 0 content",
            department="ENGINEERING",
            token_count=3,
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=1,
            content="Chunk 1 content",
            department="ENGINEERING",
            token_count=3,
        ),
    ]

    result = await repo.create_chunks(chunks)

    assert len(result) == 2
    mock_session.add_all.assert_called_once_with(chunks)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_chunk_repository_create_chunks_empty_sequence():
    """Verify create_chunks with empty sequence returns [] without calling session."""
    mock_session = AsyncMock(spec=AsyncSession)
    repo = ChunkRepository(session=mock_session)

    result = await repo.create_chunks([])
    assert result == []
    mock_session.add_all.assert_not_called()
    mock_session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunk_repository_create_chunks_with_generator():
    """Verify create_chunks handles non-list Sequence/iterables (e.g. generator) gracefully."""
    mock_session = AsyncMock(spec=AsyncSession)
    repo = ChunkRepository(session=mock_session)

    doc_id = uuid.uuid4()

    def generate_chunks():
        for i in range(3):
            yield DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc_id,
                chunk_index=i,
                content=f"Gen chunk {i}",
                department="LEGAL",
            )

    result = await repo.create_chunks(generate_chunks())
    assert len(result) == 3
    assert isinstance(result, list)
    mock_session.add_all.assert_called_once()
    mock_session.flush.assert_awaited_once()
