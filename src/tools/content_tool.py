"""
Content retrieval tool for the doc-index-mcp server.

Retrieves document content by structural location: boundary ID, chapter/section
number or title, or page range. Requires the document to be indexed.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

try:
    from ..services.chunker import BoundaryType, BoundaryIndex, Chunk, count_tokens
    from ..services.vector_store import VectorStore
except ImportError:
    from services.chunker import BoundaryType, BoundaryIndex, Chunk, count_tokens
    from services.vector_store import VectorStore


@dataclass
class ContentResponse:
    """Response from content retrieval."""
    content: str
    boundary_info: Optional[dict]  # boundary metadata if retrieved by boundary
    total_tokens: int
    chunk_count: int
    truncated: bool  # whether content was truncated due to token budget


def get_content_by_boundary(
    store: VectorStore,
    boundary_index: BoundaryIndex,
    boundary_id: str,
    include_children: bool = True,
    max_tokens: int = 8192,
) -> ContentResponse:
    """
    Retrieve content for a specific boundary ID.

    Args:
        store: Vector store containing chunks
        boundary_index: Boundary index for the document
        boundary_id: Boundary ID (e.g., "chapter:3", "section:7")
        include_children: Whether to include child boundaries
        max_tokens: Maximum tokens to return

    Returns:
        ContentResponse with the content
    """
    boundary = boundary_index.get_boundary(boundary_id)
    if not boundary:
        return ContentResponse(
            content="",
            boundary_info=None,
            total_tokens=0,
            chunk_count=0,
            truncated=False,
        )

    chunks = store.get_chunks_by_boundary(boundary_index, boundary_id, include_children)
    content, total_tokens, truncated = _assemble_content(chunks, max_tokens)

    return ContentResponse(
        content=content,
        boundary_info={
            "id": boundary.id,
            "type": boundary.type.value,
            "level": boundary.level,
            "title": boundary.title,
        },
        total_tokens=total_tokens,
        chunk_count=len(chunks),
        truncated=truncated,
    )


def get_content_by_page_range(
    store: VectorStore,
    boundary_index: BoundaryIndex,
    start_page: int,
    end_page: int,
    max_tokens: int = 8192,
) -> ContentResponse:
    """
    Retrieve content for a page range.

    Args:
        store: Vector store containing chunks
        boundary_index: Boundary index for the document
        start_page: First page (inclusive)
        end_page: Last page (inclusive)
        max_tokens: Maximum tokens to return

    Returns:
        ContentResponse with the content
    """
    # Find all chunks whose page falls within the range
    matching_chunks = []
    for chunk in store.chunks:
        page = chunk.metadata.page
        if page is not None and start_page <= page <= end_page:
            matching_chunks.append(chunk)

    # Sort by position to maintain document order
    matching_chunks.sort(key=lambda c: c.metadata.position)

    content, total_tokens, truncated = _assemble_content(matching_chunks, max_tokens)

    return ContentResponse(
        content=content,
        boundary_info={
            "type": "page_range",
            "start_page": start_page,
            "end_page": end_page,
        },
        total_tokens=total_tokens,
        chunk_count=len(matching_chunks),
        truncated=truncated,
    )


def get_content_by_title(
    store: VectorStore,
    boundary_index: BoundaryIndex,
    title_query: str,
    boundary_type: Optional[str] = None,
    include_children: bool = True,
    max_tokens: int = 8192,
) -> ContentResponse:
    """
    Retrieve content by fuzzy-matching a boundary title.

    Args:
        store: Vector store containing chunks
        boundary_index: Boundary index for the document
        title_query: Title to search for (fuzzy matched)
        boundary_type: Optional type filter ("chapter", "section", "subsection")
        include_children: Whether to include child boundaries
        max_tokens: Maximum tokens to return

    Returns:
        ContentResponse with the best-matching boundary's content
    """
    best_match = None
    best_score = 0.0

    for b in boundary_index.boundaries:
        if not b.title:
            continue

        # Filter by type if specified
        if boundary_type and b.type.value != boundary_type:
            continue

        # Skip non-structural boundaries
        if b.type not in {BoundaryType.CHAPTER, BoundaryType.SECTION, BoundaryType.SUBSECTION}:
            continue

        score = SequenceMatcher(None, title_query.lower(), b.title.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = b

    if not best_match or best_score < 0.3:
        return ContentResponse(
            content="",
            boundary_info=None,
            total_tokens=0,
            chunk_count=0,
            truncated=False,
        )

    return get_content_by_boundary(
        store, boundary_index, best_match.id, include_children, max_tokens,
    )


def get_content_by_position_range(
    store: VectorStore,
    start_position: int,
    end_position: int,
    max_tokens: int = 8192,
) -> ContentResponse:
    """
    Retrieve content by chunk position range.

    Args:
        store: Vector store containing chunks
        start_position: First chunk position (inclusive)
        end_position: Last chunk position (inclusive)
        max_tokens: Maximum tokens to return

    Returns:
        ContentResponse with the content
    """
    matching_chunks = [
        c for c in store.chunks
        if start_position <= c.metadata.position <= end_position
    ]
    matching_chunks.sort(key=lambda c: c.metadata.position)

    content, total_tokens, truncated = _assemble_content(matching_chunks, max_tokens)

    return ContentResponse(
        content=content,
        boundary_info={
            "type": "position_range",
            "start_position": start_position,
            "end_position": end_position,
        },
        total_tokens=total_tokens,
        chunk_count=len(matching_chunks),
        truncated=truncated,
    )


def _assemble_content(
    chunks: list[Chunk],
    max_tokens: int,
) -> tuple[str, int, bool]:
    """
    Assemble chunk content respecting token budget.

    Returns:
        Tuple of (content, total_tokens, truncated)
    """
    parts = []
    total_tokens = 0
    truncated = False

    for chunk in chunks:
        chunk_tokens = chunk.metadata.token_count or count_tokens(chunk.content)
        if total_tokens + chunk_tokens > max_tokens and parts:
            truncated = True
            break
        parts.append(chunk.content)
        total_tokens += chunk_tokens

    return "\n\n".join(parts), total_tokens, truncated
