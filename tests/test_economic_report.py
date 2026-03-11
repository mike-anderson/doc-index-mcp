"""Integration tests using Economic Report of the President 2024 (~500 pages, mixed content)."""

import pytest
from .conftest import run_search


@pytest.fixture
def search_ctx(economic_report_indexed, embedder):
    store, boundary_index, source = economic_report_indexed
    return {
        "stores": {source: store},
        "boundary_indices": {source: boundary_index},
        "embedder": embedder,
    }


class TestEconomicReportBoundaryDetection:
    """Verify boundary detection quality on the Economic Report."""

    async def test_not_thousands_of_chapters(self, economic_report_indexed):
        """The Economic Report has ~10 chapters, not thousands."""
        store, boundary_index, source = economic_report_indexed
        chapter_boundaries = [
            b for b in boundary_index.boundaries.values()
            if b.type.value == "chapter"
        ]
        # Should be a reasonable number, not the 4090 false positives we had before
        assert len(chapter_boundaries) < 100, (
            f"Detected {len(chapter_boundaries)} chapters — likely false positives"
        )

    async def test_boundaries_are_not_toc_entries(self, economic_report_indexed):
        """Boundaries should not be table-of-contents entries with dot-leaders."""
        store, boundary_index, source = economic_report_indexed
        for bid, b in boundary_index.boundaries.items():
            if b.title:
                assert "..." not in b.title, (
                    f"Boundary {bid} looks like a TOC entry: {b.title!r}"
                )


class TestEconomicReportBasicSearch:
    """Fact retrieval from Economic Report of the President 2024."""

    async def test_transmitted_to_congress_2024(self, search_ctx):
        """The report was transmitted to Congress in March 2024."""
        result = await run_search(
            **search_ctx,
            query="transmitted to Congress March 2024",
            top_k=3,
        )
        all_text = " ".join(r.content for r in result.results).lower()
        assert "congress" in all_text or "transmitted" in all_text

    async def test_finds_ai_chapter(self, search_ctx):
        """Chapter 7 covers AI economics."""
        result = await run_search(
            **search_ctx,
            query="economic framework artificial intelligence",
            top_k=5,
        )
        all_text = " ".join(r.content for r in result.results).lower()
        assert "artificial intelligence" in all_text or "ai" in all_text

    async def test_finds_housing_chapter(self, search_ctx):
        """Chapter 4 covers affordable housing supply."""
        result = await run_search(
            **search_ctx,
            query="increasing supply affordable housing",
            top_k=5,
        )
        all_text = " ".join(r.content for r in result.results).lower()
        assert "housing" in all_text

    async def test_finds_statistical_tables(self, search_ctx):
        """Contains statistical appendix tables with historical data."""
        result = await run_search(
            **search_ctx,
            query="statistical tables farm income",
            top_k=5,
        )
        all_text = " ".join(r.content for r in result.results).lower()
        # Should find something about tables or statistics or farm/agriculture
        assert any(w in all_text for w in ["table", "statistic", "farm", "agriculture"])


class TestEconomicReportBoundaryExpansion:
    """Test boundary features on a mixed-content document."""

    async def test_chapter_expansion_on_ai_topic(self, search_ctx):
        """Expanding to chapter on AI should return content with boundary info."""
        result = await run_search(
            **search_ctx,
            query="artificial intelligence economic impact",
            top_k=3,
            expand_to_boundary="chapter",
            max_return_tokens=16384,
        )
        assert len(result.results) > 0
        assert result.expansion_applied == "chapter"
        # At least one result should have expanded content
        has_expansion = any(r.expanded_content for r in result.results)
        assert has_expansion, "At least one result should have expanded content"

    async def test_section_expansion_returns_content(self, search_ctx):
        """Section expansion should return content with boundary metadata."""
        result = await run_search(
            **search_ctx,
            query="gross domestic product GDP growth",
            top_k=3,
            expand_to_boundary="section",
            max_return_tokens=8192,
        )
        assert len(result.results) > 0
        assert result.expansion_applied == "section"
        # Check that boundary info is populated
        items_with_boundary = [r for r in result.results if r.boundary_info]
        assert len(items_with_boundary) > 0, "Should have boundary info on expanded results"


class TestCrossDocumentSearch:
    """Test searching across multiple indexed documents."""

    async def test_search_all_three_docs(
        self, nist_csf_indexed, nist_800_53_indexed, economic_report_indexed, embedder
    ):
        """Searching across all docs should return results from relevant sources."""
        csf_store, csf_bi, csf_src = nist_csf_indexed
        nist_store, nist_bi, nist_src = nist_800_53_indexed
        econ_store, econ_bi, econ_src = economic_report_indexed

        stores = {csf_src: csf_store, nist_src: nist_store, econ_src: econ_store}
        bis = {csf_src: csf_bi, nist_src: nist_bi, econ_src: econ_bi}

        result = await run_search(
            stores=stores,
            boundary_indices=bis,
            embedder=embedder,
            query="cybersecurity risk management framework",
            top_k=5,
        )
        sources_found = {r.source_name for r in result.results}
        # Should find results from at least the NIST docs
        assert len(sources_found) >= 1
        assert any("nist" in s for s in sources_found), (
            f"Expected NIST sources in results, got: {sources_found}"
        )

    async def test_source_filter(
        self, nist_csf_indexed, nist_800_53_indexed, embedder
    ):
        """Source filter should limit results to specified source."""
        csf_store, csf_bi, csf_src = nist_csf_indexed
        nist_store, nist_bi, nist_src = nist_800_53_indexed

        stores = {csf_src: csf_store, nist_src: nist_store}
        bis = {csf_src: csf_bi, nist_src: nist_bi}

        result = await run_search(
            stores=stores,
            boundary_indices=bis,
            embedder=embedder,
            query="access control",
            top_k=5,
            sources=[nist_src],
        )
        for r in result.results:
            assert r.source_name == nist_src, (
                f"Expected only {nist_src} results, got {r.source_name}"
            )
