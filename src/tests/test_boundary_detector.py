"""
Tests for boundary detection.

Tests pattern matching, hierarchy building, boundary assignment,
and false-positive rejection (TOC lines, noise, implausible counts).
"""

import pytest
from ..services.boundary_detector import (
    detect_boundaries,
    split_content_by_boundaries,
    get_boundary_at_offset,
    assign_boundary_to_position,
    get_level_for_boundary_type,
    _is_noise_line,
    _prune_implausible,
    DetectedBoundary,
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


class TestNoiseLineFiltering:
    """Test that TOC entries, footers, and other noise are rejected."""

    def test_dot_leader_is_noise(self):
        """Lines with dot-leaders (TOC entries) should be noise."""
        assert _is_noise_line("Chapter 1 ..................... 21")
        assert _is_noise_line("The Benefits of Full Employment ......... 21")

    def test_pipe_prefix_is_noise(self):
        """Lines starting with pipes (page footers) should be noise."""
        assert _is_noise_line("| Economic Report of the President")
        assert _is_noise_line("| Annual Report of the Council")

    def test_number_pipe_is_noise(self):
        """Lines like '42 | Chapter 3' should be noise."""
        assert _is_noise_line("42 | Chapter 3")
        assert _is_noise_line("302 | References")

    def test_normal_line_is_not_noise(self):
        """Normal text and headers should not be noise."""
        assert not _is_noise_line("1 Introduction")
        assert not _is_noise_line("Chapter 4: Results")
        assert not _is_noise_line("# Section Title")
        assert not _is_noise_line("The economy grew by 3 percent.")


class TestNumberedChapterFalsePositives:
    """Test that the numbered chapter pattern rejects common false positives."""

    def test_rejects_large_numbers(self):
        """Numbers >= 100 should not match as chapter numbers."""
        content = "302 References\n\nSome text.\n\n2024 Economic Report\n\nMore text."
        boundaries = detect_boundaries(content)
        chapter_boundaries = [b for b in boundaries if b.type == BoundaryType.CHAPTER]
        # Neither "302" nor "2024" should be detected as chapters
        for b in chapter_boundaries:
            assert b.title not in ("References", "Economic Report")

    def test_rejects_toc_lines(self):
        """TOC lines with dot-leaders should not become boundaries."""
        content = """Table of Contents

1 Introduction ..................... 1
2 Methods .......................... 15
3 Results .......................... 42

1 Introduction

Actual chapter content here.
"""
        boundaries = detect_boundaries(content)
        chapter_boundaries = [b for b in boundaries if b.type == BoundaryType.CHAPTER]
        # Only the actual "1 Introduction" line (without dots) should match
        assert len(chapter_boundaries) == 1
        assert chapter_boundaries[0].title == "Introduction"

    def test_rejects_sentence_starting_with_number(self):
        """Body text lines that start with a number should not match."""
        content = "11 The composition of the workforce is known to have important implications for the broader economy.\n\nMore text."
        boundaries = detect_boundaries(content)
        chapter_boundaries = [b for b in boundaries if b.type == BoundaryType.CHAPTER]
        assert len(chapter_boundaries) == 0

    def test_accepts_legitimate_numbered_chapters(self):
        """Real numbered chapter titles should still be detected."""
        content = """1 Introduction

Content of intro.

2 Background

Content of background.

3 Methods and Data Collection

Content of methods.
"""
        boundaries = detect_boundaries(content)
        chapter_boundaries = [b for b in boundaries if b.type == BoundaryType.CHAPTER]
        assert len(chapter_boundaries) == 3
        assert chapter_boundaries[0].title == "Introduction"
        assert chapter_boundaries[1].title == "Background"
        assert chapter_boundaries[2].title == "Methods and Data Collection"

    def test_accepts_long_chapter_titles(self):
        """Chapter titles up to ~70 chars should be accepted."""
        content = "7 An Economic Framework for Understanding Artificial Intelligence\n\nContent."
        boundaries = detect_boundaries(content)
        chapter_boundaries = [b for b in boundaries if b.type == BoundaryType.CHAPTER]
        assert len(chapter_boundaries) == 1


class TestImplausiblePruning:
    """Test statistical pruning of implausible boundary counts."""

    def test_prune_excessive_chapters(self):
        """If > 100 chapters are detected, the type should be pruned entirely."""
        detected = [
            DetectedBoundary(
                boundary_type=BoundaryType.CHAPTER,
                level=1,
                title=f"Chapter {i}",
                start_offset=i * 100,
                end_offset=i * 100 + 50,
                line_number=i,
                matched_text=f"{i} Chapter {i}",
            )
            for i in range(150)
        ]
        pruned = _prune_implausible(detected, content_length=100000)
        assert len(pruned) == 0

    def test_keep_reasonable_chapters(self):
        """Reasonable chapter counts should be kept."""
        detected = [
            DetectedBoundary(
                boundary_type=BoundaryType.CHAPTER,
                level=1,
                title=f"Chapter {i}",
                start_offset=i * 100,
                end_offset=i * 100 + 50,
                line_number=i,
                matched_text=f"{i} Chapter {i}",
            )
            for i in range(20)
        ]
        pruned = _prune_implausible(detected, content_length=100000)
        assert len(pruned) == 20

    def test_prune_only_noisy_type(self):
        """Only the noisy type should be pruned, not others."""
        chapters = [
            DetectedBoundary(
                boundary_type=BoundaryType.CHAPTER,
                level=1,
                title=f"Ch {i}",
                start_offset=i * 100,
                end_offset=i * 100 + 50,
                line_number=i,
                matched_text=f"{i} Ch {i}",
            )
            for i in range(150)  # Over limit → pruned
        ]
        sections = [
            DetectedBoundary(
                boundary_type=BoundaryType.SECTION,
                level=2,
                title=f"Sec {i}",
                start_offset=i * 100 + 60,
                end_offset=i * 100 + 90,
                line_number=i,
                matched_text=f"## Sec {i}",
            )
            for i in range(10)  # Under limit → kept
        ]
        pruned = _prune_implausible(chapters + sections, content_length=100000)
        assert len(pruned) == 10
        assert all(d.boundary_type == BoundaryType.SECTION for d in pruned)
