import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ChunkResult(BaseModel):
    """
    Structured chunk entity representing a portion of a document with character offsets,
    estimated token counts, and contextual metadata (e.g. Markdown headers).
    """
    chunk_index: int
    text: str = Field(default="", description="The text content of the chunk")
    token_count: int = Field(default=0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")
    start_char: int = Field(..., description="Starting character offset in the source document")
    end_char: int = Field(..., description="Ending character offset in the source document")

    @model_validator(mode="before")
    @classmethod
    def populate_text_from_content(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "content" in data and "text" not in data:
                data["text"] = data["content"]
        return data

    @property
    def content(self) -> str:
        """Backward-compatible property returning the chunk text."""
        return self.text


class TextChunker:
    """
    Production-ready document chunker supporting recursive character splitting
    and Markdown header-aware splitting with character offset tracking and overlap preservation.
    """

    DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators is not None else list(self.DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> List[str]:
        """
        Recursively splits text using the configured hierarchy of separators.
        Guarantees that individual chunks do not exceed `chunk_size` (unless single tokens are indivisible)
        and preserves separator delimiters between joined segments.
        """
        if not text or not text.strip():
            return []

        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]

        # Find the first matching separator in the hierarchy
        selected_sep: Optional[str] = None
        remaining_seps: List[str] = []

        for idx, sep in enumerate(separators):
            if sep == "":
                selected_sep = ""
                remaining_seps = []
                break
            if sep in text:
                selected_sep = sep
                remaining_seps = separators[idx + 1:]
                break

        # Fallback to direct character slicing if no separator matched or sep is ""
        if selected_sep is None or selected_sep == "":
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [
                text[i : i + self.chunk_size].strip()
                for i in range(0, len(text), step)
                if text[i : i + self.chunk_size].strip()
            ]

        # Split text by the selected separator
        splits = text.split(selected_sep)
        chunks: List[str] = []
        current_segments: List[str] = []
        current_len = 0

        for segment in splits:
            if not segment:
                continue

            # If segment alone exceeds chunk_size, recursively split it
            if len(segment) > self.chunk_size:
                if current_segments:
                    joined = selected_sep.join(current_segments).strip()
                    if joined:
                        chunks.append(joined)
                    current_segments = []
                    current_len = 0

                sub_chunks = self._split_text(segment, remaining_seps)
                chunks.extend(sub_chunks)
                continue

            # Determine length when joined with active separator
            sep_len = len(selected_sep) if current_segments else 0
            if current_len + sep_len + len(segment) > self.chunk_size:
                if current_segments:
                    joined = selected_sep.join(current_segments).strip()
                    if joined:
                        chunks.append(joined)

                # Build overlap segments from the end of current_segments
                overlap_segments: List[str] = []
                overlap_len = 0
                for seg in reversed(current_segments):
                    needed = len(seg) + (len(selected_sep) if overlap_segments else 0)
                    if overlap_len + needed <= self.chunk_overlap:
                        overlap_segments.insert(0, seg)
                        overlap_len += needed
                    else:
                        break

                # Ensure overlap + new segment does not exceed chunk_size
                while overlap_segments:
                    tentative_len = (
                        sum(len(s) for s in overlap_segments)
                        + len(selected_sep) * len(overlap_segments)
                        + len(segment)
                    )
                    if tentative_len > self.chunk_size:
                        overlap_segments.pop(0)
                    else:
                        break

                current_segments = overlap_segments + [segment]
                current_len = (
                    sum(len(s) for s in current_segments)
                    + len(selected_sep) * (len(current_segments) - 1)
                )
            else:
                current_segments.append(segment)
                current_len += sep_len + len(segment)

        if current_segments:
            joined = selected_sep.join(current_segments).strip()
            if joined and (not chunks or joined != chunks[-1]):
                chunks.append(joined)

        return chunks

    def split_text_with_metadata(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """
        Splits text into chunks and returns structured ChunkResult instances
        with accurate start_char, end_char, chunk_index, token_count, and metadata.
        """
        if not text or not text.strip():
            return []

        meta: Dict[str, Any] = {}
        if metadata is not None:
            meta.update(metadata)
        if document_metadata is not None:
            meta.update(document_metadata)

        raw_chunks = self.split_text(text)
        if not raw_chunks:
            return []

        results: List[ChunkResult] = []
        cursor = 0
        prev_start = -1

        for idx, chunk_text in enumerate(raw_chunks):
            # Advance search cursor from prev_start + 1 to avoid overlap search traps
            search_from = max(0, prev_start + 1)
            start_pos = text.find(chunk_text, search_from)
            if start_pos == -1:
                start_pos = text.find(chunk_text, max(0, cursor - self.chunk_size))
            if start_pos == -1:
                start_pos = text.find(chunk_text)
            if start_pos == -1:
                start_pos = cursor

            end_pos = start_pos + len(chunk_text)
            prev_start = start_pos
            cursor = max(cursor, start_pos + 1)

            token_count = max(1, len(chunk_text) // 4)
            results.append(
                ChunkResult(
                    chunk_index=idx,
                    text=chunk_text,
                    token_count=token_count,
                    metadata=dict(meta),
                    start_char=start_pos,
                    end_char=end_pos,
                )
            )

        return results

    def split_markdown(
        self,
        markdown_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """
        Splits Markdown document aware of header hierarchies (#, ##, ###) and code blocks.
        Preserves preamble text, avoids parsing comments in code fences as headers,
        and accurately offsets chunks to the root document.
        """
        if not markdown_text or not markdown_text.strip():
            return []

        base_metadata: Dict[str, Any] = {}
        if metadata is not None:
            base_metadata.update(metadata)
        if document_metadata is not None:
            base_metadata.update(document_metadata)

        lines = markdown_text.splitlines(keepends=True)
        in_code_block = False

        sections: List[Dict[str, Any]] = []
        current_section_lines: List[str] = []
        current_section_offset = 0
        current_headers: Dict[str, str] = {}

        line_offset = 0
        header_regex = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in lines:
            stripped = line.strip()

            # Guard code fences
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_section_lines.append(line)
                line_offset += len(line)
                continue

            if not in_code_block:
                match = header_regex.match(stripped)
                if match:
                    # Flush previous section if it has non-empty text
                    if current_section_lines:
                        sec_text = "".join(current_section_lines)
                        if sec_text.strip():
                            sections.append({
                                "text": sec_text,
                                "offset": current_section_offset,
                                "headers": dict(current_headers),
                            })

                    # Update header hierarchy
                    hashes, title = match.group(1), match.group(2).strip()
                    level = len(hashes)
                    if level == 1:
                        current_headers = {"h1": title}
                    elif level == 2:
                        new_h: Dict[str, str] = {}
                        if "h1" in current_headers:
                            new_h["h1"] = current_headers["h1"]
                        new_h["h2"] = title
                        current_headers = new_h
                    elif level == 3:
                        new_h = {}
                        if "h1" in current_headers:
                            new_h["h1"] = current_headers["h1"]
                        if "h2" in current_headers:
                            new_h["h2"] = current_headers["h2"]
                        new_h["h3"] = title
                        current_headers = new_h

                    current_section_lines = [line]
                    current_section_offset = line_offset
                    line_offset += len(line)
                    continue

            current_section_lines.append(line)
            line_offset += len(line)

        # Flush final section
        if current_section_lines:
            sec_text = "".join(current_section_lines)
            if sec_text.strip():
                sections.append({
                    "text": sec_text,
                    "offset": current_section_offset,
                    "headers": dict(current_headers),
                })

        if not sections:
            return []

        results: List[ChunkResult] = []
        chunk_index = 0

        for sec in sections:
            sec_raw = sec["text"]
            sec_offset = sec["offset"]
            sec_headers = sec["headers"]

            # Calculate exact start accounting for leading whitespace in section
            leading_ws = len(sec_raw) - len(sec_raw.lstrip())
            sec_stripped = sec_raw.strip()
            if not sec_stripped:
                continue

            exact_start = sec_offset + leading_ws
            exact_end = exact_start + len(sec_stripped)

            if len(sec_stripped) <= self.chunk_size:
                meta = dict(base_metadata)
                meta["headers"] = dict(sec_headers)
                results.append(
                    ChunkResult(
                        chunk_index=chunk_index,
                        text=sec_stripped,
                        token_count=max(1, len(sec_stripped) // 4),
                        metadata=meta,
                        start_char=exact_start,
                        end_char=exact_end,
                    )
                )
                chunk_index += 1
            else:
                # Recursively chunk oversized section
                sub_chunks = self.split_text_with_metadata(sec_stripped, metadata=base_metadata)
                for sub_chunk in sub_chunks:
                    # Ponytail Section Offset Fix: add exact_start to sub-chunk offsets
                    sub_chunk.start_char += exact_start
                    sub_chunk.end_char += exact_start
                    sub_chunk.metadata["headers"] = dict(sec_headers)
                    sub_chunk.chunk_index = chunk_index
                    chunk_index += 1
                    results.append(sub_chunk)

        return results
