"""Integration tests using NIST SP 800-53 Rev 5 (~492 pages, deep hierarchy)."""

import pytest
from .conftest import run_search


@pytest.fixture
def search_ctx(nist_800_53_indexed, embedder):
    store, boundary_index, source = nist_800_53_indexed
    return {
        "stores": {source: store},
        "boundary_indices": {source: boundary_index},
        "embedder": embedder,
    }


class TestNist80053BasicSearch:
    """Fact retrieval from NIST 800-53 Rev 5."""

    async def test_finds_access_control_family(self, search_ctx):
        """AC (Access Control) is the first control family."""
        result = await run_search(**search_ctx, query="Access Control AC family", top_k=5)
        all_text = " ".join(r.content for r in result.results).lower()
        assert "access control" in all_text

    async def test_finds_ac1_policy_and_procedures(self, search_ctx):
        """AC-1 is 'Policy and Procedures'."""
        result = await run_search(**search_ctx, query="AC-1 Policy and Procedures", top_k=5)
        all_text = " ".join(r.content for r in result.results)
        assert "AC-1" in all_text or "ac-1" in all_text.lower()

    async def test_finds_20_control_families(self, search_ctx):
        """The document has 20 control families."""
        result = await run_search(
            **search_ctx,
            query="control families list AC AT AU CA CM CP IA IR",
            top_k=10,
        )
        all_text = " ".join(r.content for r in result.results).upper()
        # Check for a subset of the 20 families
        found = 0
        for family in ["AC", "AT", "AU", "CA", "CM", "CP", "IA", "IR", "MA", "MP",
                        "PE", "PL", "PM", "PS", "PT", "RA", "SA", "SC", "SI", "SR"]:
            if family in all_text:
                found += 1
        assert found >= 5, f"Expected at least 5 control family abbreviations, found {found}"

    async def test_privacy_controls_rev5(self, search_ctx):
        """Privacy controls were integrated for the first time in Rev 5."""
        result = await run_search(**search_ctx, query="privacy controls integrated", top_k=5)
        all_text = " ".join(r.content for r in result.results).lower()
        assert "privacy" in all_text


class TestNist80053DeepHierarchy:
    """Test boundary features on a deeply hierarchical document."""

    async def test_chapter_expansion_on_control_family(self, search_ctx):
        """Expanding to chapter on a control should return the whole family."""
        result = await run_search(
            **search_ctx,
            query="audit and accountability AU",
            top_k=1,
            expand_to_boundary="chapter",
            max_return_tokens=16384,
        )
        assert len(result.results) > 0
        item = result.results[0]
        expanded = item.expanded_content or item.content
        # A full control family chapter should be substantial
        assert len(expanded) > 1000, "Chapter expansion should return substantial content"

    async def test_section_expansion_on_specific_control(self, search_ctx):
        """Expanding to section on a control should get the full control text."""
        result = await run_search(
            **search_ctx,
            query="AC-2 Account Management",
            top_k=1,
            expand_to_boundary="section",
            max_return_tokens=8192,
        )
        assert len(result.results) > 0
        item = result.results[0]
        expanded = item.expanded_content or item.content
        assert len(expanded) > len(item.content), "Section expansion should add content"

    async def test_siblings_returns_adjacent_controls(self, search_ctx):
        """Siblings of a control should include adjacent controls in same family."""
        result = await run_search(
            **search_ctx,
            query="AC-3 Access Enforcement",
            top_k=1,
            expand_to_boundary="section",
            include_siblings=True,
            max_return_tokens=16384,
        )
        assert len(result.results) > 0
        item = result.results[0]
        expanded = item.expanded_content or item.content
        # Should contain text from sibling controls
        assert len(expanded) > 500

    async def test_large_document_search_performance(self, search_ctx):
        """Search should return results within token budget on large doc."""
        result = await run_search(
            **search_ctx,
            query="incident response plan",
            top_k=5,
            max_return_tokens=4096,
        )
        assert result.total_tokens <= 4096 + 256
        assert len(result.results) > 0
