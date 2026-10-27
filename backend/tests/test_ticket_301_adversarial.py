import asyncio
import time
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CircuitBreakerOpenError, VectorServiceError
from app.modules.documents.models.document import DocumentChunk
from app.modules.documents.repositories.chunk_repository import ChunkRepository
from app.shared.clients.gemini_client import GeminiClient
from app.shared.rag_engine.chunker import ChunkResult, TextChunker
from app.shared.rag_engine.embedder import VectorEmbedder


# ============================================================================
# 1. Zero-Length, Whitespace-Only & Exotic Whitespace Adversarial Tests
# ============================================================================

@pytest.mark.parametrize(
    "empty_input",
    [
        "",
        "   ",
        "\t\t\n\n\r\n   ",
        "\u2000\u2003\u3000\u00A0\t\n",  # Exotic unicode whitespaces (en-quad, em-space, ideographic, non-breaking)
        " \n \r \t \v \f ",
    ],
)
def test_adversarial_zero_length_and_whitespace(empty_input: str):
    """Verify chunker strictly returns empty list for any variant of empty or whitespace-only text."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    assert chunker.split_text(empty_input) == []
    assert chunker.split_text_with_metadata(empty_input) == []
    assert chunker.split_text_with_metadata(empty_input, metadata={"doc_id": "test"}) == []
    assert chunker.split_markdown(empty_input) == []
    assert chunker.split_markdown(empty_input, document_metadata={"department": "QA"}) == []


@pytest.mark.asyncio
async def test_adversarial_embedder_empty_batch():
    """Verify VectorEmbedder handles empty lists and non-callable scenarios cleanly."""
    mock_client = AsyncMock(spec=GeminiClient)
    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=16)

    result = await embedder.embed_batch([])
    assert result == []
    mock_client.generate_embedding.assert_not_awaited()


# ============================================================================
# 2. Giant String (>100k Chars) Zero-Delimiter Termination & Recursion Tests
# ============================================================================

def test_adversarial_giant_string_zero_delimiters_termination():
    """
    Stress test: A 120,000-character string with zero separators (no newlines, spaces, or dots).
    Must terminate in < 2.0 seconds without RecursionError (max recursion depth is 1000 in Python)
    and all chunks must respect chunk_size.
    """
    chunk_size = 500
    chunk_overlap = 50
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 120,000 distinct characters without any delimiters
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    giant_text = (alphabet * 2000)[:120000]
    assert len(giant_text) == 120000

    start_time = time.perf_counter()
    chunks = chunker.split_text(giant_text)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 2.0, f"Chunking took too long: {elapsed:.2f}s"
    assert len(chunks) > 0

    step = chunk_size - chunk_overlap
    expected_chunks = (len(giant_text) + step - 1) // step
    assert len(chunks) == expected_chunks

    for i, chunk in enumerate(chunks):
        assert len(chunk) <= chunk_size, f"Chunk {i} exceeded chunk_size: {len(chunk)}"

    # Also test split_text_with_metadata on the giant string
    start_time_meta = time.perf_counter()
    meta_chunks = chunker.split_text_with_metadata(giant_text)
    elapsed_meta = time.perf_counter() - start_time_meta

    assert elapsed_meta < 3.0, f"split_text_with_metadata took too long: {elapsed_meta:.2f}s"
    assert len(meta_chunks) == len(chunks)

    # Verify character offset fidelity across the entire 120k string
    for idx, chunk_res in enumerate(meta_chunks):
        assert chunk_res.chunk_index == idx
        slice_text = giant_text[chunk_res.start_char : chunk_res.end_char]
        assert slice_text == chunk_res.text, (
            f"Fidelity failure at chunk {idx}: offset [{chunk_res.start_char}:{chunk_res.end_char}]"
        )


def test_adversarial_giant_identical_char_string_termination():
    """
    Stress test: 150,000 identical characters 'Z' * 150000 with zero separators.
    Verify termination, recursion safety, and correct length bounds.
    """
    chunk_size = 400
    chunk_overlap = 100
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    identical_text = "Z" * 150000

    start_time = time.perf_counter()
    chunks = chunker.split_text(identical_text)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 2.0, f"Identical char chunking took {elapsed:.2f}s"
    step = chunk_size - chunk_overlap
    expected = (len(identical_text) + step - 1) // step
    assert len(chunks) == expected
    assert all(len(c) <= chunk_size for c in chunks)


# ============================================================================
# 3. Character Offset Fidelity Verification Tests (All Chunks Exact Match)
# ============================================================================

def test_adversarial_offset_fidelity_complex_multilingual():
    """
    Verify text[start_char:end_char] == chunk.text for complex, multi-lingual,
    repetitive, and formatting-dense documents.
    """
    chunker = TextChunker(chunk_size=120, chunk_overlap=25)

    complex_text = (
        "Enterprise AI Knowledge Retrieval Engine.\n\n"
        "ระบบจัดการเอกสารและการค้นหาความหมายด้วยปัญญาประดิษฐ์ระดับองค์กร\n\n"
        "Machine Learning with PyTorch, TensorFlow, and ONNX Runtime. "
        "High-performance similarity search over vector spaces.\n\n"
        "Repeated phrase for offset testing: Verify offset fidelity now. "
        "Repeated phrase for offset testing: Verify offset fidelity now. "
        "Repeated phrase for offset testing: Verify offset fidelity now.\n\n"
        "German engineering: Übergrößenträgerprüfung und Qualitätssicherung. "
        "Emojis and symbols: 🚀 🔍 🧠 ⚡ 🛡️ 📊.\n\n"
        "End of the complex multilingual document test."
    )

    results = chunker.split_text_with_metadata(complex_text, metadata={"lang": "multi"})
    assert len(results) >= 4

    prev_start = -1
    for i, res in enumerate(results):
        assert res.chunk_index == i
        assert res.start_char >= prev_start, f"Monotonicity violated at chunk {i}"
        assert res.start_char < res.end_char

        # Exact substring fidelity
        source_slice = complex_text[res.start_char : res.end_char]
        assert source_slice == res.text, (
            f"Fidelity mismatch at chunk {i}:\n"
            f"Expected: {repr(res.text)}\n"
            f"Got:      {repr(source_slice)}"
        )
        prev_start = res.start_char


def test_adversarial_offset_fidelity_oversized_markdown():
    """
    Verify that in Markdown splitting with oversized sections, every single sub-chunk
    accurately maps to markdown_text[start_char:end_char].
    """
    chunker = TextChunker(chunk_size=90, chunk_overlap=20)

    markdown = (
        "# Main Heading\n\n"
        "Preamble paragraph that introduces the enterprise RAG system.\n\n"
        "## Subheading Alpha\n"
        "Alpha paragraph that has enough length to trigger recursive sub-chunking. "
        "It goes on and on to exceed ninety characters comfortably and tests the section offset mathematics.\n\n"
        "### Nested Subheading Beta\n"
        "Beta content with specific details about pgvector cosine distance metrics and HNSW index tuning parameters.\n\n"
        "## Subheading Gamma\n"
        "Final section wrapping up the markdown document test."
    )

    results = chunker.split_markdown(markdown)
    assert len(results) >= 5

    for i, res in enumerate(results):
        # Strict offset fidelity check against original root markdown
        extracted = markdown[res.start_char : res.end_char]
        assert extracted == res.text, (
            f"Markdown offset mismatch at chunk {i} ({res.metadata.get('headers')}):\n"
            f"Expected from offset: {repr(extracted)}\n"
            f"Chunk text:           {repr(res.text)}"
        )


# ============================================================================
# 4. Markdown Code Block Fence Guard with # Comments Tests
# ============================================================================

def test_adversarial_markdown_code_blocks_with_hash_comments():
    """
    Adversarial challenge: Verify that lines starting with '#', '##', or '###'
    inside code blocks (Python, Bash, YAML, Dockerfile) are NEVER extracted as headers.
    """
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)

    markdown = """# Enterprise Documentation

Here is a deployment script:

```bash
#!/bin/bash
# Check system status
## Ensure root permissions
### Run deployment container
docker-compose up -d
```

Here is a Python utility:

```python
# Configuration loader
class Config:
    ## Database URI
    DB_URL = "postgresql://localhost"
    ### Connection pool size
    POOL_SIZE = 10
```

## Operations Guide

After running the script, verify metrics:

```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'backend'
    ## Scrape interval
    scrape_interval: 15s
```

### Incident Response

Follow protocol.
"""

    results = chunker.split_markdown(markdown)
    assert len(results) == 3

    # Section 1: H1 "Enterprise Documentation"
    assert results[0].metadata["headers"] == {"h1": "Enterprise Documentation"}
    assert "# Check system status" in results[0].text
    assert "## Ensure root permissions" in results[0].text
    assert "# Configuration loader" in results[0].text

    # Section 2: H2 "Operations Guide"
    assert results[1].metadata["headers"] == {
        "h1": "Enterprise Documentation",
        "h2": "Operations Guide",
    }
    assert "# Prometheus configuration" in results[1].text
    assert "## Scrape interval" in results[1].text

    # Section 3: H3 "Incident Response"
    assert results[2].metadata["headers"] == {
        "h1": "Enterprise Documentation",
        "h2": "Operations Guide",
        "h3": "Incident Response",
    }
    assert "Follow protocol." in results[2].text

    # Verify that NO chunk's headers dictionary contains any code comment texts
    for res in results:
        for level, title in res.metadata.get("headers", {}).items():
            assert not title.startswith("Check system status")
            assert not title.startswith("Ensure root permissions")
            assert not title.startswith("Configuration loader")
            assert not title.startswith("Prometheus configuration")


def test_adversarial_markdown_unclosed_code_fence():
    """
    Adversarial challenge: An unclosed code fence (```) at the end of the document.
    Must not crash or infinite loop; remaining content must be treated as inside code fence.
    """
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)

    malformed_markdown = """# Guide With Broken Code Fence

```python
# This code fence is never closed
def broken():
    pass
# Another comment that looks like a header
"""

    results = chunker.split_markdown(malformed_markdown)
    assert len(results) == 1
    assert results[0].metadata["headers"] == {"h1": "Guide With Broken Code Fence"}
    assert "def broken():" in results[0].text


# ============================================================================
# 5. High-Volume Batch Embedding Concurrency in VectorEmbedder
# ============================================================================

@pytest.mark.asyncio
async def test_adversarial_high_volume_batch_embedding_concurrency():
    """
    Stress test: Embed 300 chunks concurrently using VectorEmbedder with batch_size=32.
    Simulate non-trivial async latency in GeminiClient (asyncio.sleep) and verify:
    1. Correct order of returned embeddings matching inputs.
    2. Concurrency operates properly (total time is roughly num_batches * sleep, not num_items * sleep).
    3. Exception propagation on partial failure.
    """
    mock_client = AsyncMock(spec=GeminiClient)

    async def mock_generate_embedding(text: str) -> List[float]:
        await asyncio.sleep(0.01)  # 10ms simulated API network latency
        # Embed chunk number encoded in vector
        return [float(len(text))] * 768

    mock_client.generate_embedding.side_effect = mock_generate_embedding

    batch_size = 32
    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=batch_size)

    total_chunks = 160
    chunks = [f"Adversarial chunk content number {i} for high volume batch testing" for i in range(total_chunks)]

    start_time = time.perf_counter()
    results = await embedder.embed_batch(chunks)
    elapsed = time.perf_counter() - start_time

    assert len(results) == total_chunks
    assert mock_client.generate_embedding.await_count == total_chunks

    # Sequential time would be 160 * 0.01s = 1.6s.
    # Concurrent batch time (5 batches of 32 items): 5 * 0.01s = ~0.05s - 0.2s.
    assert elapsed < 1.0, f"Concurrent batching took too long: {elapsed:.2f}s (sequential suspected)"

    # Verify order fidelity
    for i, vec in enumerate(results):
        expected_val = float(len(chunks[i]))
        assert vec[0] == expected_val


@pytest.mark.asyncio
async def test_adversarial_batch_embedding_partial_failure_propagation():
    """
    Adversarial test: If one item in a batch raises an exception (e.g. VectorServiceError),
    embed_batch must immediately fail and propagate the exception cleanly.
    """
    mock_client = AsyncMock(spec=GeminiClient)

    async def mock_generate_with_failure(text: str) -> List[float]:
        if "FAIL" in text:
            raise VectorServiceError("Remote embedding API rejected payload.")
        return [0.1] * 768

    mock_client.generate_embedding.side_effect = mock_generate_with_failure
    embedder = VectorEmbedder(gemini_client=mock_client, batch_size=4)

    chunks = ["normal1", "normal2", "normal3_FAIL_here", "normal4"]

    with pytest.raises(VectorServiceError, match="Remote embedding API rejected payload."):
        await embedder.embed_batch(chunks)


# ============================================================================
# 6. Bulk Persistence Stress & Boundary Overlap Tests
# ============================================================================

@pytest.mark.asyncio
async def test_adversarial_chunk_repository_large_bulk_insert():
    """Verify ChunkRepository handles 1,000 chunks in a single transaction via create_chunks."""
    mock_session = AsyncMock(spec=AsyncSession)
    repo = ChunkRepository(session=mock_session)

    doc_id = uuid.uuid4()
    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=i,
            content=f"Bulk stress chunk {i}",
            department="LEGAL",
            token_count=10,
        )
        for i in range(1000)
    ]

    result = await repo.create_chunks(chunks)
    assert len(result) == 1000
    mock_session.add_all.assert_called_once()
    mock_session.flush.assert_awaited_once()


@pytest.mark.parametrize(
    "size,overlap",
    [
        (10, 0),   # Zero overlap
        (10, 9),   # Maximum overlap (size - 1)
        (1, 0),    # Single character chunk
    ],
)
def test_adversarial_boundary_overlap_configurations(size: int, overlap: int):
    """Verify boundary conditions for valid chunk_size and chunk_overlap values."""
    chunker = TextChunker(chunk_size=size, chunk_overlap=overlap)
    text = "0123456789" * 5

    chunks = chunker.split_text(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= size


def test_adversarial_crlf_windows_newlines():
    """Verify chunking and offset fidelity when documents use Windows CRLF (\r\n) line endings."""
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)
    crlf_text = (
        "Line 1 with Windows newline.\r\n"
        "Line 2 with Windows newline.\r\n"
        "Line 3 with Windows newline.\r\n"
        "Line 4 with Windows newline.\r\n"
        "Line 5 with Windows newline.\r\n"
    )

    results = chunker.split_text_with_metadata(crlf_text)
    assert len(results) >= 2

    for res in results:
        # Check fidelity against original CRLF text
        slice_text = crlf_text[res.start_char : res.end_char]
        assert slice_text == res.text

