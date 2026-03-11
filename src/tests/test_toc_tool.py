"""Tests for the TOC tool."""

import pytest
from ..services.chunker import Boundary, BoundaryType, BoundaryIndex
from ..tools.toc_tool import build_toc, toc_to_dict


def _make_boundary_index():
    """Create a BoundaryIndex with a realistic hierarchy."""
    bi = BoundaryIndex()

    # Chapter 1 with 2 sections
    bi.add_boundary(Boundary(
        type=BoundaryType.CHAPTER, level=1, id="chapter:1",
        title="Introduction", start_offset=0,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.SECTION, level=2, id="section:1",
        title="Background", start_offset=100, parent_id="chapter:1",
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.SECTION, level=2, id="section:2",
        title="Motivation", start_offset=500, parent_id="chapter:1",
    ))

    # Chapter 2 with a section and subsection
    bi.add_boundary(Boundary(
        type=BoundaryType.CHAPTER, level=1, id="chapter:2",
        title="Methods", start_offset=1000,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.SECTION, level=2, id="section:3",
        title="Data Collection", start_offset=1100, parent_id="chapter:2",
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.SUBSECTION, level=3, id="subsection:1",
        title="Survey Design", start_offset=1200, parent_id="section:3",
    ))

    # Page boundaries (should be excluded from TOC)
    bi.add_boundary(Boundary(
        type=BoundaryType.PAGE, level=4, id="page:1",
        title="1", start_offset=0,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.PAGE, level=4, id="page:2",
        title="2", start_offset=500,
    ))

    # Map some chunks
    for i in range(3):
        bi.map_chunk_to_boundary(f"doc:{i}", "chapter:1")
    for i in range(3, 5):
        bi.map_chunk_to_boundary(f"doc:{i}", "section:1")
    for i in range(5, 7):
        bi.map_chunk_to_boundary(f"doc:{i}", "section:2")
    for i in range(7, 10):
        bi.map_chunk_to_boundary(f"doc:{i}", "chapter:2")
    for i in range(10, 12):
        bi.map_chunk_to_boundary(f"doc:{i}", "section:3")
    bi.map_chunk_to_boundary("doc:12", "subsection:1")

    return bi


class TestBuildToc:
    def test_returns_chapters(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=1)
        assert len(toc) == 2
        assert toc[0].title == "Introduction"
        assert toc[1].title == "Methods"

    def test_excludes_pages(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=4)
        all_types = set()
        def collect_types(entries):
            for e in entries:
                all_types.add(e.type)
                collect_types(e.children)
        collect_types(toc)
        assert "page" not in all_types

    def test_nested_sections(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=2)
        ch1 = toc[0]
        assert len(ch1.children) == 2
        assert ch1.children[0].title == "Background"
        assert ch1.children[1].title == "Motivation"

    def test_nested_subsections(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=3)
        ch2 = toc[1]
        assert len(ch2.children) == 1
        section3 = ch2.children[0]
        assert len(section3.children) == 1
        assert section3.children[0].title == "Survey Design"

    def test_max_depth_limits(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=2)
        ch2 = toc[1]
        section3 = ch2.children[0]
        # Subsection should be excluded at depth 2
        assert len(section3.children) == 0

    def test_chunk_counts(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=3)
        ch1 = toc[0]
        # chapter:1 has 3 direct chunks, sections add 4 more = 7 total
        assert ch1.chunk_count == 3
        assert ch1.total_chunks == 7

    def test_empty_index(self):
        bi = BoundaryIndex()
        toc = build_toc(bi)
        assert toc == []


class TestTocToDict:
    def test_serialization(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=2)
        result = toc_to_dict(toc)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Introduction"
        assert result[0]["boundary_id"] == "chapter:1"
        assert "children" in result[0]
        assert len(result[0]["children"]) == 2

    def test_no_children_key_when_empty(self):
        bi = _make_boundary_index()
        toc = build_toc(bi, max_depth=1)
        result = toc_to_dict(toc)
        # Chapters with no children at depth=1 shouldn't have "children" key
        assert "children" not in result[0]
