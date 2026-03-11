"""
Table of contents tool for the mcp-knowledge MCP server.

Extracts hierarchical document structure (chapters, sections, subsections)
from indexed documents or by scanning unindexed documents on-the-fly.
"""

from dataclasses import dataclass
from typing import Optional

try:
    from ..services.chunker import BoundaryType, BoundaryIndex, Boundary
except ImportError:
    from services.chunker import BoundaryType, BoundaryIndex, Boundary


# Boundary types that represent document structure (not layout)
STRUCTURAL_TYPES = {
    BoundaryType.CHAPTER,
    BoundaryType.SECTION,
    BoundaryType.SUBSECTION,
}


@dataclass
class TocEntry:
    """A single entry in the table of contents."""
    boundary_id: str
    type: str           # "chapter", "section", "subsection"
    level: int          # 1-4
    title: Optional[str]
    chunk_count: int    # Number of chunks in this boundary (direct only)
    total_chunks: int   # Including children
    children: list["TocEntry"]
    page: Optional[int] = None


def build_toc(
    boundary_index: BoundaryIndex,
    max_depth: int = 3,
) -> list[TocEntry]:
    """
    Build a hierarchical table of contents from a BoundaryIndex.

    Args:
        boundary_index: The boundary index for the document
        max_depth: Maximum depth level to include (1=chapters only,
                   2=+sections, 3=+subsections)

    Returns:
        List of top-level TocEntry objects with nested children
    """
    # Filter to structural boundaries only, within max_depth
    structural = [
        b for b in boundary_index.boundaries
        if b.type in STRUCTURAL_TYPES and b.level <= max_depth
    ]

    if not structural:
        return []

    # Build a map of boundary_id -> TocEntry
    entries: dict[str, TocEntry] = {}
    for b in structural:
        direct_chunks = boundary_index.boundary_to_chunks.get(b.id, [])
        all_chunks = boundary_index.get_chunks_in_boundary(b.id, include_children=True)

        # Try to find page number from chunks
        page = _find_page_for_boundary(b, boundary_index)

        entries[b.id] = TocEntry(
            boundary_id=b.id,
            type=b.type.value,
            level=b.level,
            title=b.title,
            chunk_count=len(direct_chunks),
            total_chunks=len(all_chunks),
            children=[],
            page=page,
        )

    # Build hierarchy
    roots: list[TocEntry] = []
    for b in structural:
        entry = entries[b.id]
        if b.parent_id and b.parent_id in entries:
            entries[b.parent_id].children.append(entry)
        else:
            roots.append(entry)

    return roots


def _find_page_for_boundary(
    boundary: Boundary,
    boundary_index: BoundaryIndex,
) -> Optional[int]:
    """Find the page number for a boundary by looking at page boundaries."""
    best_page = None
    best_offset = -1
    for b in boundary_index.boundaries:
        if b.type == BoundaryType.PAGE and b.start_offset <= boundary.start_offset:
            if b.start_offset > best_offset:
                best_offset = b.start_offset
                try:
                    best_page = int(b.title)
                except (ValueError, TypeError):
                    pass
    return best_page


def toc_to_dict(entries: list[TocEntry]) -> list[dict]:
    """Convert TocEntry tree to a serializable dict list."""
    result = []
    for entry in entries:
        d = {
            "boundary_id": entry.boundary_id,
            "type": entry.type,
            "level": entry.level,
            "title": entry.title,
            "chunk_count": entry.chunk_count,
            "total_chunks": entry.total_chunks,
        }
        if entry.page is not None:
            d["page"] = entry.page
        if entry.children:
            d["children"] = toc_to_dict(entry.children)
        result.append(d)
    return result
