"""
Tests for boundary detection.

Tests pattern matching, hierarchy building, and boundary assignment.
"""

import pytest
from ..services.boundary_detector import (
    detect_boundaries,
    split_content_by_boundaries,
    get_boundary_at_offset,
    assign_boundary_to_position,
    get_level_for_boundary_type,
)
from ..services.chunker import BoundaryType


class TestMarkdownBoundaryDetection:
    """Test detection of markdown headers."""

    def test_detect_h1_chapter(self):
        """Should detect # as chapter boundary."""
        content = "# Introduction\n\nSome content here."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.CHAPTER
        assert boundaries[0].level == 1
        assert boundaries[0].title == "Introduction"

    def test_detect_h2_section(self):
        """Should detect ## as section boundary."""
        content = "## Methods\n\nMethodology description."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.SECTION
        assert boundaries[0].level == 2
        assert boundaries[0].title == "Methods"

    def test_detect_h3_subsection(self):
        """Should detect ### as subsection boundary."""
        content = "### Data Collection\n\nCollection details."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.SUBSECTION
        assert boundaries[0].level == 3
        assert boundaries[0].title == "Data Collection"

    def test_detect_nested_headers(self):
        """Should detect nested markdown headers with hierarchy."""
        content = """# Chapter 1

Introduction text.

## Section 1.1

Section content.

### Subsection 1.1.1

Subsection content.

## Section 1.2

More section content.
"""
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 4

        # Chapter
        assert boundaries[0].type == BoundaryType.CHAPTER
        assert boundaries[0].level == 1
        assert boundaries[0].parent_id is None

        # Section 1.1
        assert boundaries[1].type == BoundaryType.SECTION
        assert boundaries[1].level == 2
        assert boundaries[1].parent_id == boundaries[0].id

        # Subsection 1.1.1
        assert boundaries[2].type == BoundaryType.SUBSECTION
        assert boundaries[2].level == 3
        assert boundaries[2].parent_id == boundaries[1].id

        # Section 1.2
        assert boundaries[3].type == BoundaryType.SECTION
        assert boundaries[3].level == 2
        assert boundaries[3].parent_id == boundaries[0].id


class TestPageBoundaryDetection:
    """Test detection of page markers."""

    def test_detect_page_marker_brackets(self):
        """Should detect [Page N] format."""
        content = "[Page 1]\n\nPage one content.\n\n[Page 2]\n\nPage two content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 2
        assert all(b.type == BoundaryType.PAGE for b in boundaries)
        assert boundaries[0].title == "1"
        assert boundaries[1].title == "2"

    def test_detect_page_marker_dashes(self):
        """Should detect --- Page N --- format."""
        content = "--- Page 1 ---\n\nContent.\n\n--- Page 2 ---\n\nMore content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 2
        assert all(b.type == BoundaryType.PAGE for b in boundaries)


class TestTraditionalChapterDetection:
    """Test detection of traditional chapter markers."""

    def test_detect_chapter_keyword(self):
        """Should detect 'Chapter X' format."""
        content = "Chapter 1: Introduction\n\nChapter content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.CHAPTER
        assert boundaries[0].level == 1

    def test_detect_roman_numeral_chapter(self):
        """Should detect 'Chapter IV' format."""
        content = "Chapter IV: The Climax\n\nDramatic content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.CHAPTER


class TestNumberedSectionDetection:
    """Test detection of numbered sections."""

    def test_detect_simple_numbered_section(self):
        """Should detect '1 Title' format."""
        content = "1 Introduction\n\nIntro content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.CHAPTER
        assert boundaries[0].level == 1
        assert boundaries[0].title == "Introduction"

    def test_detect_subsection_numbers(self):
        """Should detect '1.2 Title' format."""
        content = "1.2 Background\n\nBackground info."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.SECTION
        assert boundaries[0].level == 2

    def test_detect_subsubsection_numbers(self):
        """Should detect '1.2.3 Title' format."""
        content = "1.2.3 Implementation Details\n\nDetails here."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.SUBSECTION
        assert boundaries[0].level == 3


class TestAllCapsDetection:
    """Test detection of ALL CAPS headers."""

    def test_detect_all_caps_header(self):
        """Should detect ALL CAPS section headers."""
        content = "INTRODUCTION\n\nIntro content."
        boundaries = detect_boundaries(content)

        assert len(boundaries) == 1
        assert boundaries[0].type == BoundaryType.SECTION
        assert boundaries[0].title == "INTRODUCTION"

    def test_ignore_short_all_caps(self):
        """Should not detect short ALL CAPS (< 4 chars)."""
        content = "THE\n\nSome content."
        boundaries = detect_boundaries(content)

        # 'THE' is too short
        assert len(boundaries) == 0


class TestHierarchyBuilding:
    """Test hierarchy relationship building."""

    def test_parent_child_relationships(self):
        """Should build correct parent-child relationships."""
        content = """# Chapter

## Section A

### Subsection A1

### Subsection A2

## Section B

### Subsection B1
"""
        boundaries = detect_boundaries(content)

        # Build a lookup for easier testing
        by_title = {b.title: b for b in boundaries}

        assert by_title["Section A"].parent_id == by_title["Chapter"].id
        assert by_title["Section B"].parent_id == by_title["Chapter"].id
        assert by_title["Subsection A1"].parent_id == by_title["Section A"].id
        assert by_title["Subsection A2"].parent_id == by_title["Section A"].id
        assert by_title["Subsection B1"].parent_id == by_title["Section B"].id


class TestContentSplitting:
    """Test content splitting by boundaries."""

    def test_split_by_boundaries(self):
        """Should split content into regions by boundary."""
        content = """# Chapter 1

First chapter content.

## Section 1

Section content.

## Section 2

More section content.
"""
        boundaries = detect_boundaries(content)
        regions = split_content_by_boundaries(content, boundaries)

        assert len(regions) == 3  # Chapter + 2 sections

        # Each region should have content and a boundary
        for region_content, boundary in regions:
            assert len(region_content) > 0
            assert boundary is not None

    def test_preamble_content(self):
        """Should capture content before first boundary."""
        content = """Preamble content here.

# Chapter 1

Chapter content.
"""
        boundaries = detect_boundaries(content)
        regions = split_content_by_boundaries(content, boundaries)

        # First region should be preamble with no boundary
        assert regions[0][1] is None
        assert "Preamble" in regions[0][0]


class TestBoundaryAtOffset:
    """Test finding boundary at character offset."""

    def test_find_boundary_at_offset(self):
        """Should find correct boundary for given offset."""
        content = """# Chapter 1

Content in chapter.

## Section 1

Content in section.
"""
        boundaries = detect_boundaries(content)

        # Offset in section content should find section
        section_offset = content.index("Content in section")
        boundary = get_boundary_at_offset(boundaries, section_offset)

        assert boundary is not None
        assert boundary.type == BoundaryType.SECTION

    def test_offset_before_any_boundary(self):
        """Should return None for offset before any boundary."""
        content = """Preamble.

# Chapter 1

Content.
"""
        boundaries = detect_boundaries(content)
        boundary = get_boundary_at_offset(boundaries, 0)

        # No boundary contains offset 0
        assert boundary is None


class TestLevelMapping:
    """Test boundary type to level mapping."""

    def test_level_for_chapter(self):
        assert get_level_for_boundary_type("chapter") == 1

    def test_level_for_section(self):
        assert get_level_for_boundary_type("section") == 2

    def test_level_for_subsection(self):
        assert get_level_for_boundary_type("subsection") == 3

    def test_level_for_page(self):
        assert get_level_for_boundary_type("page") == 4

    def test_level_for_unknown(self):
        # Unknown types default to subsection level
        assert get_level_for_boundary_type("unknown") == 3
