"""
Boundary detection service for document structure analysis.

Detects logical boundaries (chapters, sections, headers) in documents
to enable boundary-aware chunking and retrieval.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .chunker import Boundary, BoundaryType


@dataclass
class BoundaryPattern:
    """A pattern for detecting document boundaries."""
    pattern: re.Pattern
    boundary_type: BoundaryType
    level: int
    title_group: int = 1  # Which regex group contains the title
    priority: int = 0     # Higher priority patterns are checked first


# Pattern definitions ordered by priority (higher priority checked first)
BOUNDARY_PATTERNS: list[BoundaryPattern] = [
    # Markdown headers (most common in our context)
    BoundaryPattern(
        pattern=re.compile(r'^#\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.CHAPTER,
        level=1,
        title_group=1,
        priority=100,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^##\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.SECTION,
        level=2,
        title_group=1,
        priority=99,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^###\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.SUBSECTION,
        level=3,
        title_group=1,
        priority=98,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^####\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.SUBSECTION,
        level=4,
        title_group=1,
        priority=97,
    ),

    # Page markers (common in PDF extractions)
    BoundaryPattern(
        pattern=re.compile(r'^\[Page\s+(\d+)\]$', re.MULTILINE | re.IGNORECASE),
        boundary_type=BoundaryType.PAGE,
        level=4,
        title_group=1,
        priority=90,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^---\s*Page\s+(\d+)\s*---$', re.MULTILINE | re.IGNORECASE),
        boundary_type=BoundaryType.PAGE,
        level=4,
        title_group=1,
        priority=89,
    ),

    # Traditional chapter markers
    BoundaryPattern(
        pattern=re.compile(r'^(?:CHAPTER|Chapter)\s+(\d+|[IVXLC]+)(?:[:.]?\s+(.+))?$', re.MULTILINE),
        boundary_type=BoundaryType.CHAPTER,
        level=1,
        title_group=2,  # The actual title is in group 2
        priority=85,
    ),

    # Numbered sections (e.g., "1.2.3 Title")
    BoundaryPattern(
        pattern=re.compile(r'^(\d+)\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.CHAPTER,
        level=1,
        title_group=2,
        priority=70,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^(\d+\.\d+)\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.SECTION,
        level=2,
        title_group=2,
        priority=71,
    ),
    BoundaryPattern(
        pattern=re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)$', re.MULTILINE),
        boundary_type=BoundaryType.SUBSECTION,
        level=3,
        title_group=2,
        priority=72,
    ),

    # ALL CAPS headers (common in legal/academic docs)
    BoundaryPattern(
        pattern=re.compile(r'^([A-Z][A-Z\s]{3,}[A-Z])$', re.MULTILINE),
        boundary_type=BoundaryType.SECTION,
        level=2,
        title_group=1,
        priority=50,
    ),
]


@dataclass
class DetectedBoundary:
    """A boundary detected in text with its match information."""
    boundary_type: BoundaryType
    level: int
    title: Optional[str]
    start_offset: int
    end_offset: int
    line_number: int
    matched_text: str


def detect_boundaries(content: str) -> list[Boundary]:
    """
    Detect logical boundaries in document content.

    Scans the document line by line, testing each against boundary patterns.
    Maintains a hierarchy stack to assign parent relationships.

    Args:
        content: Document text content

    Returns:
        List of Boundary objects with hierarchy relationships
    """
    # First, find all potential boundaries
    detected: list[DetectedBoundary] = []

    lines = content.split('\n')
    offset = 0

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            offset += len(line) + 1
            continue

        # Test against patterns in priority order
        sorted_patterns = sorted(BOUNDARY_PATTERNS, key=lambda p: -p.priority)

        for pattern in sorted_patterns:
            match = pattern.pattern.match(stripped)
            if match:
                # Extract title from the appropriate group
                title = None
                if pattern.title_group <= len(match.groups()):
                    title = match.group(pattern.title_group)
                    if title:
                        title = title.strip()

                # For numbered sections, include the number in ID generation
                detected.append(DetectedBoundary(
                    boundary_type=pattern.boundary_type,
                    level=pattern.level,
                    title=title,
                    start_offset=offset,
                    end_offset=offset + len(line),
                    line_number=line_num,
                    matched_text=stripped,
                ))
                break  # Only match first pattern per line

        offset += len(line) + 1

    # Build boundaries with hierarchy
    return _build_hierarchy(detected)


def _build_hierarchy(detected: list[DetectedBoundary]) -> list[Boundary]:
    """
    Build Boundary objects with parent relationships from detected boundaries.

    Uses a stack-based approach to track the current hierarchy context.
    Each boundary's parent is the most recent boundary at a higher level.
    """
    boundaries: list[Boundary] = []
    # Stack of (level, boundary_id) for tracking hierarchy
    level_stack: list[tuple[int, str]] = []
    # Counters for generating unique IDs per type
    type_counters: dict[str, int] = {}

    for detected_boundary in detected:
        # Generate unique ID
        type_key = detected_boundary.boundary_type.value
        if type_key not in type_counters:
            type_counters[type_key] = 0
        type_counters[type_key] += 1
        boundary_id = f"{type_key}:{type_counters[type_key]}"

        # Find parent by popping stack until we find a higher level
        parent_id = None
        while level_stack and level_stack[-1][0] >= detected_boundary.level:
            level_stack.pop()

        if level_stack:
            parent_id = level_stack[-1][1]

        # Create boundary
        boundary = Boundary(
            type=detected_boundary.boundary_type,
            level=detected_boundary.level,
            id=boundary_id,
            title=detected_boundary.title,
            start_offset=detected_boundary.start_offset,
            parent_id=parent_id,
        )
        boundaries.append(boundary)

        # Push to stack
        level_stack.append((detected_boundary.level, boundary_id))

    return boundaries


def get_boundary_at_offset(boundaries: list[Boundary], offset: int) -> Optional[Boundary]:
    """
    Find the most specific boundary containing a given character offset.

    Returns the boundary with the highest level (most specific) that
    starts before the given offset.

    Args:
        boundaries: List of boundaries to search
        offset: Character offset in the document

    Returns:
        The most specific containing boundary, or None if no boundary contains the offset
    """
    containing = [b for b in boundaries if b.start_offset <= offset]
    if not containing:
        return None

    # Sort by start_offset descending, then by level descending (most specific)
    containing.sort(key=lambda b: (-b.start_offset, -b.level))
    return containing[0]


def assign_boundary_to_position(
    boundaries: list[Boundary],
    start_offset: int,
    end_offset: int
) -> Optional[Boundary]:
    """
    Determine which boundary a text span belongs to.

    A span belongs to the boundary that:
    1. Starts at or before the span's start
    2. Is the most specific (highest level) such boundary

    Args:
        boundaries: List of detected boundaries
        start_offset: Start character offset of the span
        end_offset: End character offset of the span

    Returns:
        The boundary this span belongs to, or None
    """
    return get_boundary_at_offset(boundaries, start_offset)


def split_content_by_boundaries(
    content: str,
    boundaries: list[Boundary]
) -> list[tuple[str, Optional[Boundary]]]:
    """
    Split content into regions based on detected boundaries.

    Each region is the text from one boundary to the next (or end of document).
    Content before the first boundary is returned with boundary=None.

    Args:
        content: Full document content
        boundaries: Detected boundaries sorted by start_offset

    Returns:
        List of (region_content, boundary) tuples
    """
    if not boundaries:
        return [(content, None)]

    # Sort boundaries by start_offset
    sorted_boundaries = sorted(boundaries, key=lambda b: b.start_offset)

    regions: list[tuple[str, Optional[Boundary]]] = []

    # Content before first boundary
    if sorted_boundaries[0].start_offset > 0:
        preamble = content[:sorted_boundaries[0].start_offset].strip()
        if preamble:
            regions.append((preamble, None))

    # Content for each boundary region
    for i, boundary in enumerate(sorted_boundaries):
        start = boundary.start_offset

        # Find end (next boundary start or end of document)
        if i + 1 < len(sorted_boundaries):
            end = sorted_boundaries[i + 1].start_offset
        else:
            end = len(content)

        region_content = content[start:end].strip()
        if region_content:
            regions.append((region_content, boundary))

    return regions


def get_level_for_boundary_type(type_name: str) -> int:
    """Get the hierarchical level for a boundary type name."""
    level_map = {
        "document": 0,
        "chapter": 1,
        "section": 2,
        "subsection": 3,
        "page": 4,
    }
    return level_map.get(type_name, 3)
