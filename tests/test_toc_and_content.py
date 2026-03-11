"""Integration tests for knowledge_toc and knowledge_get_content tools."""

import pytest
from .conftest import run_search
from src.tools.toc_tool import build_toc, toc_to_dict
from src.tools.content_tool import (
    get_content_by_boundary,
    get_content_by_page_range,
    get_content_by_title,
)


@pytest.fixture
def nist_csf_ctx(nist_csf_indexed):
    store, boundary_index, source = nist_csf_indexed
    return store, boundary_index, source


@pytest.fixture
def nist_800_53_ctx(nist_800_53_indexed):
    store, boundary_index, source = nist_800_53_indexed
    return store, boundary_index, source


@pytest.fixture
def economic_report_ctx(economic_report_indexed):
    store, boundary_index, source = economic_report_indexed
    return store, boundary_index, source


class TestTocNistCsf:
    """TOC extraction from NIST CSF 2.0 (~32 pages)."""

    def test_has_structural_entries(self, nist_csf_ctx):
        store, bi, source = nist_csf_ctx
        toc = build_toc(bi, max_depth=3)
        assert len(toc) > 0, "Should have at least one top-level entry"

    def test_toc_serializes(self, nist_csf_ctx):
        store, bi, source = nist_csf_ctx
        toc = build_toc(bi, max_depth=2)
        result = toc_to_dict(toc)
        assert isinstance(result, list)
        for entry in result:
            assert "boundary_id" in entry
            assert "type" in entry
            assert "title" in entry


class TestTocNist80053:
    """TOC extraction from NIST 800-53 (~492 pages, deep hierarchy)."""

    def test_has_chapters(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        toc = build_toc(bi, max_depth=1)
        assert len(toc) > 0, "Should detect chapters in NIST 800-53"

    def test_has_sections_under_chapters(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        toc = build_toc(bi, max_depth=2)
        has_children = any(len(entry.children) > 0 for entry in toc)
        assert has_children, "At least one chapter should have sections"

    def test_reasonable_entry_count(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        toc = build_toc(bi, max_depth=3)
        total = sum(1 for _ in _flatten(toc))
        # Should have a reasonable number of entries, not thousands
        assert total < 500, f"Too many TOC entries: {total}"


class TestTocEconomicReport:
    """TOC extraction from Economic Report (~500 pages)."""

    def test_has_chapters(self, economic_report_ctx):
        store, bi, source = economic_report_ctx
        toc = build_toc(bi, max_depth=1)
        assert len(toc) > 0, "Should detect chapters"

    def test_reasonable_chapter_count(self, economic_report_ctx):
        store, bi, source = economic_report_ctx
        toc = build_toc(bi, max_depth=1)
        # Economic report has ~10 chapters
        assert len(toc) < 50, f"Too many chapters: {len(toc)}"


class TestGetContentNistCsf:
    """Content retrieval from NIST CSF 2.0."""

    def test_get_first_boundary(self, nist_csf_ctx):
        store, bi, source = nist_csf_ctx
        toc = build_toc(bi, max_depth=1)
        if not toc:
            pytest.skip("No TOC entries found")
        first_id = toc[0].boundary_id
        resp = get_content_by_boundary(store, bi, first_id)
        assert resp.chunk_count > 0
        assert len(resp.content) > 100

    def test_get_by_page_range(self, nist_csf_ctx):
        store, bi, source = nist_csf_ctx
        resp = get_content_by_page_range(store, bi, 1, 3)
        assert resp.chunk_count > 0

    def test_token_budget_respected(self, nist_csf_ctx):
        store, bi, source = nist_csf_ctx
        resp = get_content_by_page_range(store, bi, 1, 100, max_tokens=2048)
        assert resp.total_tokens <= 2048 + 256  # small buffer for last chunk


class TestGetContentNist80053:
    """Content retrieval from NIST 800-53."""

    def test_get_chapter_by_boundary_id(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        toc = build_toc(bi, max_depth=1)
        if not toc:
            pytest.skip("No chapters found")
        resp = get_content_by_boundary(store, bi, toc[0].boundary_id)
        assert resp.chunk_count > 0

    def test_get_by_title(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        resp = get_content_by_title(store, bi, "Access Control")
        assert resp.chunk_count > 0
        assert resp.content  # Non-empty

    def test_get_page_range(self, nist_800_53_ctx):
        store, bi, source = nist_800_53_ctx
        resp = get_content_by_page_range(store, bi, 10, 15, max_tokens=8192)
        assert resp.chunk_count > 0


class TestGetContentEconomicReport:
    """Content retrieval from Economic Report."""

    def test_get_by_title_housing(self, economic_report_ctx):
        store, bi, source = economic_report_ctx
        resp = get_content_by_title(store, bi, "housing")
        # May or may not find an exact match, but shouldn't error
        if resp.chunk_count > 0:
            assert "housing" in resp.content.lower() or resp.boundary_info is not None

    def test_get_early_pages(self, economic_report_ctx):
        store, bi, source = economic_report_ctx
        resp = get_content_by_page_range(store, bi, 1, 5)
        assert resp.chunk_count > 0

    def test_truncation_on_large_range(self, economic_report_ctx):
        store, bi, source = economic_report_ctx
        resp = get_content_by_page_range(store, bi, 1, 500, max_tokens=4096)
        assert resp.truncated is True
        assert resp.total_tokens <= 4096 + 256


def _flatten(entries):
    """Flatten a TocEntry tree into an iterator."""
    for entry in entries:
        yield entry
        yield from _flatten(entry.children)
