"""
Chunking service for the doc-index-mcp server.

Provides boundary-aware document chunking optimized for embedding models
with limited context windows (e.g., 256 tokens for all-MiniLM-L6-v2).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import re
import tiktoken


class BoundaryType(Enum):
    """Types of logical boundaries in documents."""
    DOCUMENT = "document"
    CHAPTER = "chapter"       # # in MD, Chapter X
    SECTION = "section"       # ## in MD, 1.1
    SUBSECTION = "subsection" # ###, 1.1.1
    PAGE = "page"             # [Page N]
    SHEET = "sheet"           # Excel sheet
    SLIDE = "slide"           # PPTX slide
    ROW_GROUP = "row_group"   # Excel row range within a sheet


@dataclass
class Boundary:
    """Represents a logical boundary in a document."""
    type: BoundaryType
    level: int               # 1-4 (1=highest/document, 4=lowest/page)
    id: str                  # e.g., "chapter:1", "section:2"
    title: Optional[str]
    start_offset: int        # Character offset in original document
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "level": self.level,
            "id": self.id,
            "title": self.title,
            "start_offset": self.start_offset,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Boundary":
        return cls(
            type=BoundaryType(data["type"]),
            level=data["level"],
            id=data["id"],
            title=data["title"],
            start_offset=data["start_offset"],
            parent_id=data.get("parent_id"),
        )


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk with boundary support."""
    source: str
    position: int
    total_chunks: int
    page: Optional[int] = None
    section: Optional[str] = None
    boundary_type: Optional[str] = None
    boundary_id: Optional[str] = None
    boundary_level: Optional[int] = None
    parent_boundary: Optional[str] = None
    boundary_title: Optional[str] = None
    token_count: Optional[int] = None
    sheet: Optional[str] = None
    slide: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "source": self.source,
            "position": self.position,
            "total_chunks": self.total_chunks,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.section is not None:
            result["section"] = self.section
        if self.sheet is not None:
            result["sheet"] = self.sheet
        if self.slide is not None:
            result["slide"] = self.slide
        if self.boundary_type is not None:
            result["boundary_type"] = self.boundary_type
        if self.boundary_id is not None:
            result["boundary_id"] = self.boundary_id
        if self.boundary_level is not None:
            result["boundary_level"] = self.boundary_level
        if self.parent_boundary is not None:
            result["parent_boundary"] = self.parent_boundary
        if self.boundary_title is not None:
            result["boundary_title"] = self.boundary_title
        if self.token_count is not None:
            result["token_count"] = self.token_count
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        return cls(
            source=data["source"],
            position=data["position"],
            total_chunks=data["total_chunks"],
            page=data.get("page"),
            section=data.get("section"),
            boundary_type=data.get("boundary_type"),
            boundary_id=data.get("boundary_id"),
            boundary_level=data.get("boundary_level"),
            parent_boundary=data.get("parent_boundary"),
            boundary_title=data.get("boundary_title"),
            token_count=data.get("token_count"),
            sheet=data.get("sheet"),
            slide=data.get("slide"),
        )


@dataclass
class Chunk:
    """A document chunk with content and metadata."""
    id: str
    content: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=ChunkMetadata.from_dict(data["metadata"]),
        )


@dataclass
class ChunkOptions:
    """Configuration options for document chunking."""
    chunk_size: int = 256           # Target tokens (optimized for embedding model)
    chunk_overlap: int = 32         # Overlap tokens between chunks
    max_chunk_size: int = 1024      # Max tokens for unbroken content
    min_chunk_size: int = 64        # Minimum tokens (avoid tiny chunks)
    respect_boundaries: bool = True  # Split at logical boundaries

    def validate(self):
        """Validate options are sensible."""
        if self.chunk_size < 32:
            raise ValueError("chunk_size must be at least 32 tokens")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")


@dataclass
class BoundaryIndex:
    """Index for tracking boundaries and their relationships to chunks."""
    boundaries: list[Boundary] = field(default_factory=list)
    chunk_to_boundary: dict[str, str] = field(default_factory=dict)       # chunk_id -> boundary_id
    boundary_to_chunks: dict[str, list[str]] = field(default_factory=dict) # boundary_id -> [chunk_ids]
    hierarchy: dict[str, list[str]] = field(default_factory=dict)         # parent_id -> [child_ids]

    def add_boundary(self, boundary: Boundary):
        """Add a boundary and update hierarchy."""
        self.boundaries.append(boundary)
        if boundary.parent_id:
            if boundary.parent_id not in self.hierarchy:
                self.hierarchy[boundary.parent_id] = []
            self.hierarchy[boundary.parent_id].append(boundary.id)

    def map_chunk_to_boundary(self, chunk_id: str, boundary_id: str):
        """Associate a chunk with its containing boundary."""
        self.chunk_to_boundary[chunk_id] = boundary_id
        if boundary_id not in self.boundary_to_chunks:
            self.boundary_to_chunks[boundary_id] = []
        self.boundary_to_chunks[boundary_id].append(chunk_id)

    def get_boundary(self, boundary_id: str) -> Optional[Boundary]:
        """Get a boundary by ID."""
        for b in self.boundaries:
            if b.id == boundary_id:
                return b
        return None

    def get_chunks_in_boundary(self, boundary_id: str, include_children: bool = False) -> list[str]:
        """Get all chunk IDs within a boundary."""
        chunks = self.boundary_to_chunks.get(boundary_id, []).copy()
        if include_children:
            for child_id in self.hierarchy.get(boundary_id, []):
                chunks.extend(self.get_chunks_in_boundary(child_id, include_children=True))
        return chunks

    def get_ancestor_at_level(self, boundary_id: str, target_level: int) -> Optional[str]:
        """Walk up hierarchy to find ancestor at target level."""
        boundary = self.get_boundary(boundary_id)
        if not boundary:
            return None
        if boundary.level == target_level:
            return boundary_id
        if boundary.level < target_level:
            return None  # Already above target level
        if boundary.parent_id:
            return self.get_ancestor_at_level(boundary.parent_id, target_level)
        return None

    def get_siblings(self, boundary_id: str) -> list[str]:
        """Get sibling boundaries (same parent)."""
        boundary = self.get_boundary(boundary_id)
        if not boundary or not boundary.parent_id:
            return []
        return [
            b.id for b in self.boundaries
            if b.parent_id == boundary.parent_id and b.id != boundary_id
        ]

    def to_dict(self) -> dict:
        return {
            "boundaries": [b.to_dict() for b in self.boundaries],
            "chunk_to_boundary": self.chunk_to_boundary,
            "boundary_to_chunks": self.boundary_to_chunks,
            "hierarchy": self.hierarchy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoundaryIndex":
        index = cls(
            boundaries=[Boundary.from_dict(b) for b in data.get("boundaries", [])],
            chunk_to_boundary=data.get("chunk_to_boundary", {}),
            boundary_to_chunks=data.get("boundary_to_chunks", {}),
            hierarchy=data.get("hierarchy", {}),
        )
        return index

    def save(self, path: str):
        """Save boundary index to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BoundaryIndex":
        """Load boundary index from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Tokenizer for counting tokens
_tokenizer = None

def get_tokenizer():
    """Get or create the tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        # Use cl100k_base which is compatible with most models
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(get_tokenizer().encode(text))


# Sentence splitting patterns
_SENTENCE_END_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_ABBREVIATIONS = {'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Jr.', 'Sr.', 'Inc.', 'Ltd.', 'Corp.', 'vs.', 'etc.', 'e.g.', 'i.e.'}
_CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)


def _split_into_sentences(content: str) -> list[str]:
    """
    Split content into sentences while preserving code blocks.

    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations (Mr., Dr., etc.)
    - Code blocks (``` ... ```)

    Args:
        content: Text content to split

    Returns:
        List of sentences/segments
    """
    # Protect code blocks by replacing them with placeholders
    code_blocks: list[str] = []

    def protect_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    protected = _CODE_BLOCK_PATTERN.sub(protect_code_block, content)

    # Split on sentence boundaries
    # First, protect abbreviations
    for abbrev in _ABBREVIATIONS:
        protected = protected.replace(abbrev, abbrev.replace('.', '__DOT__'))

    # Split on sentence endings
    sentences = _SENTENCE_END_PATTERN.split(protected)

    # Restore abbreviations and code blocks
    result = []
    for sentence in sentences:
        sentence = sentence.replace('__DOT__', '.')
        # Restore code blocks
        for i, block in enumerate(code_blocks):
            sentence = sentence.replace(f"__CODE_BLOCK_{i}__", block)
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result


def _chunk_region(
    region_content: str,
    boundary: Optional[Boundary],
    source_name: str,
    chunk_offset: int,
    options: ChunkOptions
) -> list[Chunk]:
    """
    Chunk a single region (content under one boundary) into appropriately sized chunks.

    Strategy:
    1. Target 256 tokens for optimal embedding
    2. Allow up to max_chunk_size for unbroken content
    3. Never split mid-sentence
    4. Apply overlap within same boundary

    Args:
        region_content: Text content of this region
        boundary: The boundary this region belongs to (may be None for preamble)
        source_name: Source document name
        chunk_offset: Starting chunk number for ID generation
        options: Chunking configuration

    Returns:
        List of chunks for this region
    """
    # Split into sentences for fine-grained control
    sentences = _split_into_sentences(region_content)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Handle oversized sentences (e.g., code blocks)
        if sentence_tokens > options.max_chunk_size:
            # Flush current content first
            if current_sentences:
                chunk_content = ' '.join(current_sentences)
                chunks.append(_create_chunk(
                    content=chunk_content,
                    source_name=source_name,
                    position=chunk_offset + len(chunks),
                    boundary=boundary,
                    token_count=current_tokens,
                ))
                current_sentences = []
                current_tokens = 0

            # Add oversized sentence as its own chunk
            chunks.append(_create_chunk(
                content=sentence,
                source_name=source_name,
                position=chunk_offset + len(chunks),
                boundary=boundary,
                token_count=sentence_tokens,
            ))
            continue

        # Check if adding this sentence exceeds target
        if current_tokens + sentence_tokens > options.chunk_size and current_sentences:
            # Create chunk with current content
            chunk_content = ' '.join(current_sentences)
            chunks.append(_create_chunk(
                content=chunk_content,
                source_name=source_name,
                position=chunk_offset + len(chunks),
                boundary=boundary,
                token_count=current_tokens,
            ))

            # Start new chunk - apply overlap within boundary
            overlap_sentences = _get_overlap_sentences(current_sentences, options.chunk_overlap)
            current_sentences = overlap_sentences + [sentence]
            current_tokens = sum(count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Add final chunk if content remains
    if current_sentences:
        chunk_content = ' '.join(current_sentences)
        token_count = count_tokens(chunk_content)

        # Only create chunk if it meets minimum size (or it's all we have)
        if token_count >= options.min_chunk_size or not chunks:
            chunks.append(_create_chunk(
                content=chunk_content,
                source_name=source_name,
                position=chunk_offset + len(chunks),
                boundary=boundary,
                token_count=token_count,
            ))
        elif chunks:
            # Merge tiny trailing content with previous chunk
            prev_chunk = chunks[-1]
            prev_chunk.content = prev_chunk.content + ' ' + chunk_content
            prev_chunk.metadata.token_count = count_tokens(prev_chunk.content)

    return chunks


def _get_overlap_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Get sentences from the end that total approximately overlap_tokens."""
    if overlap_tokens <= 0 or not sentences:
        return []

    result = []
    total_tokens = 0

    for sentence in reversed(sentences):
        tokens = count_tokens(sentence)
        if total_tokens + tokens > overlap_tokens and result:
            break
        result.insert(0, sentence)
        total_tokens += tokens

    return result


def _create_chunk(
    content: str,
    source_name: str,
    position: int,
    boundary: Optional[Boundary],
    token_count: int,
) -> Chunk:
    """Create a Chunk with proper metadata."""
    metadata = ChunkMetadata(
        source=source_name,
        position=position,
        total_chunks=0,  # Updated later
        token_count=token_count,
    )

    if boundary:
        metadata.boundary_type = boundary.type.value
        metadata.boundary_id = boundary.id
        metadata.boundary_level = boundary.level
        metadata.parent_boundary = boundary.parent_id
        metadata.boundary_title = boundary.title

    return Chunk(
        id=f"{source_name}:{position}",
        content=content,
        metadata=metadata,
    )


def chunk_document(
    content: str,
    source_name: str,
    options: Optional[ChunkOptions] = None,
    loader_boundaries: Optional[list[Boundary]] = None,
) -> tuple[list[Chunk], BoundaryIndex]:
    """
    Boundary-aware chunking algorithm.

    Two-phase strategy:
    1. Detect logical boundaries (chapters, sections, headers)
       - Merges with pre-detected boundaries from native document loaders
    2. Chunk each region optimally for embedding model

    Optimized for:
    - 256-token target (matches all-MiniLM-L6-v2 context window)
    - Sentence-level splitting (never breaks mid-sentence)
    - Boundary preservation (metadata tracks containing boundary)

    Args:
        content: Document text content
        source_name: Name/identifier for the source document
        options: Chunking options
        loader_boundaries: Pre-detected boundaries from native document loaders

    Returns:
        Tuple of (chunks, boundary_index)
    """
    # Import here to avoid circular imports
    from .boundary_detector import (
        detect_boundaries,
        get_loader_boundary_types,
        merge_boundaries,
        split_content_by_boundaries,
    )

    if options is None:
        options = ChunkOptions()  # Uses v2 defaults

    options.validate()

    # Phase 1: Detect text-pattern boundaries
    # Skip regex detection for types the loader already provides natively —
    # the loader's boundaries are authoritative and we don't want unreliable
    # duplicate detection from pattern matching.
    skip_types = get_loader_boundary_types(loader_boundaries) if loader_boundaries else set()
    detected_boundaries = detect_boundaries(content, skip_types=skip_types)

    # Merge with loader-provided boundaries (native page/sheet/slide detection)
    if loader_boundaries:
        boundaries = merge_boundaries(loader_boundaries, detected_boundaries)
    else:
        boundaries = detected_boundaries

    # Build boundary index
    boundary_index = BoundaryIndex()
    for boundary in boundaries:
        boundary_index.add_boundary(boundary)

    # Phase 2: Split content by boundaries and chunk each region
    regions = split_content_by_boundaries(content, boundaries)

    all_chunks: list[Chunk] = []

    for region_content, boundary in regions:
        region_chunks = _chunk_region(
            region_content=region_content,
            boundary=boundary,
            source_name=source_name,
            chunk_offset=len(all_chunks),
            options=options,
        )

        # Map chunks to their boundary and populate structured metadata
        for chunk in region_chunks:
            if boundary:
                boundary_index.map_chunk_to_boundary(chunk.id, boundary.id)

                # Populate page number from page boundaries
                if boundary.type == BoundaryType.PAGE:
                    try:
                        chunk.metadata.page = int(boundary.title)
                    except (ValueError, TypeError):
                        pass
                # Populate sheet name from sheet boundaries
                elif boundary.type == BoundaryType.SHEET:
                    chunk.metadata.sheet = boundary.title
                # Populate slide number from slide boundaries
                elif boundary.type == BoundaryType.SLIDE:
                    try:
                        # Slide ID is "slide:N"
                        chunk.metadata.slide = int(boundary.id.split(":")[1])
                    except (ValueError, IndexError):
                        pass
                # For sub-boundaries, inherit from ancestors
                elif boundary.type == BoundaryType.ROW_GROUP:
                    # Row groups are children of sheets
                    if boundary.parent_id:
                        parent = boundary_index.get_boundary(boundary.parent_id)
                        if parent and parent.type == BoundaryType.SHEET:
                            chunk.metadata.sheet = parent.title

                # For non-page boundaries, find the page they're on by offset
                if boundary.type != BoundaryType.PAGE and chunk.metadata.page is None:
                    chunk.metadata.page = _find_page_at_offset(
                        boundary.start_offset, boundary_index
                    )

        all_chunks.extend(region_chunks)

    # Update total_chunks in all metadata
    for chunk in all_chunks:
        chunk.metadata.total_chunks = len(all_chunks)

    return all_chunks, boundary_index


def _find_page_at_offset(
    offset: int,
    boundary_index: BoundaryIndex,
) -> Optional[int]:
    """Find the page number for content at a given offset.

    Searches all PAGE boundaries for the nearest one that precedes the offset.
    This works regardless of hierarchy since pages are layout boundaries
    orthogonal to semantic boundaries (headings, sections).
    """
    best_page = None
    best_offset = -1
    for b in boundary_index.boundaries:
        if b.type == BoundaryType.PAGE and b.start_offset <= offset:
            if b.start_offset > best_offset:
                best_offset = b.start_offset
                try:
                    best_page = int(b.title)
                except (ValueError, TypeError):
                    pass
    return best_page
