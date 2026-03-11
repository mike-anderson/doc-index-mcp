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
    # The bare "N Title" pattern requires a small chapter number (1-99) and a
    # short, title-like remainder — no dot-leaders, pipes, or long sentences.
    BoundaryPattern(
        pattern=re.compile(r'^(\d{1,2})\s+([A-Z][A-Za-z0-9,:\-\'\(\)\s]{2,70})$', re.MULTILINE),
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

    # Sheet markers (Excel)
    BoundaryPattern(
        pattern=re.compile(r'^\[Sheet:\s+(.+)\]$', re.MULTILINE),
        boundary_type=BoundaryType.SHEET,
        level=1,
        title_group=1,
        priority=92,
    ),

    # Row group markers (Excel)
    BoundaryPattern(
        pattern=re.compile(r'^\[Rows\s+(\d+-\d+)\]$', re.MULTILINE),
        boundary_type=BoundaryType.ROW_GROUP,
        level=2,
        title_group=1,
        priority=88,
    ),

    # Slide markers (PPTX)
    BoundaryPattern(
        pattern=re.compile(r'^\[Slide\s+(\d+):\s+(.+)\]$', re.MULTILINE),
        boundary_type=BoundaryType.SLIDE,
        level=1,
        title_group=2,
        priority=91,
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


# Lines matching these patterns are never boundaries (TOC entries, page footers, etc.)
_NOISE_LINE_PATTERNS = [
    re.compile(r'\.{3,}'),           # Dot-leaders (TOC entries)
    re.compile(r'^\d+\s*\|\s'),      # Page-number pipes ("42 | Chapter 3")
    re.compile(r'^\|\s'),            # Leading pipes ("| Economic Report")
]


def _is_noise_line(line: str) -> bool:
    """Return True if the line looks like a TOC entry, footer, or other noise."""
    return any(p.search(line) for p in _NOISE_LINE_PATTERNS)


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


def detect_boundaries(
    content: str,
    skip_types: Optional[set[BoundaryType]] = None,
) -> list[Boundary]:
    """
    Detect logical boundaries in document content.

    Scans the document line by line, testing each against boundary patterns.
    Maintains a hierarchy stack to assign parent relationships.

    Args:
        content: Document text content
        skip_types: Boundary types to skip detection for (e.g., when the
            loader already provides these natively and we don't want
            unreliable duplicate detection from regex patterns)

    Returns:
        List of Boundary objects with hierarchy relationships
    """
    # First, find all potential boundaries
    detected: list[DetectedBoundary] = []
    skip_types = skip_types or set()

    lines = content.split('\n')
    offset = 0

    # Filter out patterns for types the loader already handles
    active_patterns = [p for p in BOUNDARY_PATTERNS if p.boundary_type not in skip_types]

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            offset += len(line) + 1
            continue

        # Skip lines that look like TOC entries, footers, etc.
        if _is_noise_line(stripped):
            offset += len(line) + 1
            continue

        # Test against patterns in priority order
        sorted_patterns = sorted(active_patterns, key=lambda p: -p.priority)

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

    # Validate: if a boundary type has implausibly many detections, drop it.
    # Real documents rarely have more than ~50 chapters or ~200 sections.
    detected = _prune_implausible(detected, len(content))

    # Build boundaries with hierarchy
    return _build_hierarchy(detected)


# Maximum plausible boundary counts by type.
# Real documents rarely exceed these; if they do, the pattern is matching noise.
_MAX_BOUNDARIES = {
    BoundaryType.CHAPTER: 100,
    BoundaryType.SECTION: 500,
    BoundaryType.SUBSECTION: 1000,
}


def _prune_implausible(
    detected: list[DetectedBoundary],
    content_length: int,
) -> list[DetectedBoundary]:
    """Remove boundary types that have implausibly many detections.

    If a pattern fired far more times than any real document structure would
    produce, it was matching noise (TOC lines, numbered paragraphs, etc.).
    """
    from collections import Counter
    type_counts = Counter(d.boundary_type for d in detected)

    noisy_types: set[BoundaryType] = set()
    for btype, count in type_counts.items():
        max_allowed = _MAX_BOUNDARIES.get(btype)
        if max_allowed and count > max_allowed:
            noisy_types.add(btype)

    if not noisy_types:
        return detected

    return [d for d in detected if d.boundary_type not in noisy_types]


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
        "sheet": 1,
        "slide": 1,
        "row_group": 2,
    }
    return level_map.get(type_name, 3)


def get_loader_boundary_types(loader_boundaries: list[Boundary]) -> set[BoundaryType]:
    """Extract the set of boundary types provided by the loader.

    These types should be skipped during regex detection to avoid
    unreliable duplicate boundaries.
    """
    return {b.type for b in loader_boundaries}


def merge_boundaries(
    loader_boundaries: list[Boundary],
    detected_boundaries: list[Boundary],
) -> list[Boundary]:
    """
    Merge pre-detected loader boundaries with text-pattern-detected boundaries.

    The detected_boundaries should already have loader-provided types filtered
    out (via skip_types in detect_boundaries), so this just combines them and
    parents text-detected boundaries (e.g., headings) under the nearest
    preceding loader boundary.

    Args:
        loader_boundaries: Boundaries from native document parsing
        detected_boundaries: Boundaries from regex pattern matching
            (already filtered to exclude loader-provided types)

    Returns:
        Merged boundary list, sorted by offset
    """
    if not loader_boundaries:
        return detected_boundaries
    if not detected_boundaries:
        return loader_boundaries

    merged = list(loader_boundaries)

    for detected in detected_boundaries:
        # Parent text-detected boundaries under nearest preceding loader boundary
        if detected.parent_id is None:
            nearest_parent = None
            for lb in sorted(loader_boundaries, key=lambda b: b.start_offset):
                if lb.start_offset <= detected.start_offset and lb.level < detected.level:
                    nearest_parent = lb
            if nearest_parent:
                detected.parent_id = nearest_parent.id

        merged.append(detected)

    merged.sort(key=lambda b: b.start_offset)
    return merged
