import asyncio
from typing import List, Optional
from app.shared.clients.gemini_client import GeminiClient


class VectorEmbedder:
    """
    Service to convert text chunks into vector embeddings using GeminiClient.
    Provides single-chunk and concurrent batch embedding with batch slicing.
    """

    def __init__(self, gemini_client: GeminiClient, batch_size: int = 16):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.gemini_client = gemini_client
        self.batch_size = batch_size

    async def embed_chunk(self, chunk_text: str) -> List[float]:
        """Generate 768-dim normalized embedding via GeminiClient."""
        return await self.gemini_client.generate_embedding(chunk_text)

    async def embed_batch(
        self,
        chunks: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Batch-embed multiple chunks with concurrent batch processing.
        Slices chunks into batches of size `effective_batch_size` and uses asyncio.gather.
        """
        if not chunks:
            return []

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        results: List[List[float]] = []
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i : i + effective_batch_size]
            batch_results = await asyncio.gather(
                *(self.embed_chunk(text) for text in batch)
            )
            results.extend(batch_results)

        return results
