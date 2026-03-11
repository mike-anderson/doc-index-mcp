"""Integration tests using NIST Cybersecurity Framework 2.0 (~32 pages, compact)."""

import pytest
from .conftest import run_search


@pytest.fixture
def search_ctx(nist_csf_indexed, embedder):
    store, boundary_index, source = nist_csf_indexed
    return {
        "stores": {source: store},
        "boundary_indices": {source: boundary_index},
        "embedder": embedder,
    }


class TestNistCsfBasicSearch:
    """Basic fact retrieval from NIST CSF 2.0."""

    async def test_finds_six_core_functions(self, search_ctx):
        """CSF 2.0 defines six Functions: Govern, Identify, Protect, Detect, Respond, Recover."""
        result = await run_search(**search_ctx, query="CSF core functions", top_k=5)
        all_text = " ".join(r.content for r in result.results).lower()
        for fn in ["govern", "identify", "protect", "detect", "respond", "recover"]:
            assert fn in all_text, f"Expected to find function '{fn}' in search results"

    async def test_govern_is_new_in_csf2(self, search_ctx):
        """Govern was added as a new function in CSF 2.0."""
        result = await run_search(**search_ctx, query="Govern function new in CSF 2.0", top_k=5)
        all_text = " ".join(r.content for r in result.results).lower()
        assert "govern" in all_text

    async def test_finds_implementation_tiers(self, search_ctx):
        """CSF defines four Tiers: Partial, Risk Informed, Repeatable, Adaptive."""
        result = await run_search(**search_ctx, query="implementation tiers", top_k=5)
        all_text = " ".join(r.content for r in result.results).lower()
        for tier in ["partial", "risk informed", "repeatable", "adaptive"]:
            assert tier in all_text, f"Expected to find tier '{tier}'"

    async def test_publication_date(self, search_ctx):
        """Published February 26, 2024."""
        result = await run_search(**search_ctx, query="publication date February 2024", top_k=3)
        all_text = " ".join(r.content for r in result.results)
        assert "February" in all_text or "2024" in all_text


class TestNistCsfBoundaryExpansion:
    """Test boundary expansion features on NIST CSF."""

    async def test_section_expansion_returns_more_content(self, search_ctx):
        """Expanding to section should return more content than default."""
        basic = await run_search(**search_ctx, query="CSF profiles", top_k=3)
        expanded = await run_search(
            **search_ctx,
            query="CSF profiles",
            top_k=2,
            expand_to_boundary="section",
            max_return_tokens=8192,
        )
        basic_len = sum(len(r.content) for r in basic.results)
        expanded_len = sum(
            len(r.expanded_content or r.content) for r in expanded.results
        )
        assert expanded_len > basic_len, "Section expansion should return more content"

    async def test_expansion_has_boundary_info(self, search_ctx):
        """Expanded results should include boundary metadata."""
        result = await run_search(
            **search_ctx,
            query="CSF core",
            top_k=1,
            expand_to_boundary="section",
            max_return_tokens=8192,
        )
        assert len(result.results) > 0
        item = result.results[0]
        assert item.expanded_content is not None, "Should have expanded content"
        assert item.boundary_info is not None, "Should have boundary info"

    async def test_sibling_expansion_returns_peer_sections(self, search_ctx):
        """Including siblings should return content from adjacent sections."""
        without_siblings = await run_search(
            **search_ctx,
            query="CSF tiers",
            top_k=1,
            expand_to_boundary="section",
            max_return_tokens=8192,
        )
        with_siblings = await run_search(
            **search_ctx,
            query="CSF tiers",
            top_k=1,
            expand_to_boundary="section",
            include_siblings=True,
            max_return_tokens=16384,
        )
        without_len = sum(
            len(r.expanded_content or r.content) for r in without_siblings.results
        )
        with_len = sum(
            len(r.expanded_content or r.content) for r in with_siblings.results
        )
        assert with_len >= without_len, "Siblings should add content"

    async def test_token_budget_respected(self, search_ctx):
        """Results should not exceed max_return_tokens."""
        result = await run_search(
            **search_ctx,
            query="cybersecurity",
            top_k=3,
            expand_to_boundary="section",
            max_return_tokens=4096,
        )
        assert result.total_tokens <= 4096 + 256, (
            f"Token budget exceeded: {result.total_tokens} > 4096"
        )
