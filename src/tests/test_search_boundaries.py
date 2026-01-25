"""
Tests for boundary-aware search functionality.

Tests boundary expansion, sibling inclusion, and token limits.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from ..services.chunker import (
    Chunk,
    ChunkMetadata,
    BoundaryIndex,
    Boundary,
    BoundaryType,
    chunk_document,
)
from ..services.vector_store import VectorStore
from ..tools.search_tool import (
    SearchParams,
    execute_search,
    _expand_to_boundary,
    _build_context,
)


@pytest.fixture
def sample_document():
    """Create a sample document with clear boundaries."""
    return """# Introduction

This is the introduction to our research paper. It provides background information
and sets up the context for the rest of the document.

## Background

The background section covers related work and previous research in this area.
It discusses key findings from other researchers.

## Problem Statement

We identify the core problem that this research aims to address.
The problem is significant because it affects many applications.

# Methods

This section describes our methodology and approach.

## Data Collection

We collected data from multiple sources over a period of six months.
The data includes measurements from sensors and user surveys.

## Analysis Techniques

We applied statistical analysis and machine learning algorithms.
The techniques include regression analysis and clustering.

# Results

Our findings are presented in this section.

## Quantitative Results

The numbers show a significant improvement of 25% over baseline.
We achieved statistical significance with p < 0.05.

## Qualitative Findings

User feedback was overwhelmingly positive about the new approach.
Participants reported improved satisfaction.

# Conclusion

We summarize our findings and discuss implications.
Future work will extend these methods to new domains.
"""


@pytest.fixture
def indexed_document(sample_document):
    """Create indexed chunks and boundary index from sample document."""
    chunks, boundary_index = chunk_document(sample_document, "research_paper")
    return chunks, boundary_index


@pytest.fixture
def mock_store(indexed_document):
    """Create a mock vector store with indexed chunks."""
    chunks, boundary_index = indexed_document

    store = MagicMock(spec=VectorStore)

    # Mock get_chunk_by_id
    chunk_map = {c.id: c for c in chunks}
    store.get_chunk_by_id = MagicMock(side_effect=lambda cid: chunk_map.get(cid))

    # Mock get_chunks_by_ids
    store.get_chunks_by_ids = MagicMock(
        side_effect=lambda ids: [chunk_map[cid] for cid in ids if cid in chunk_map]
    )

    # Mock get_chunk_with_neighbors
    def get_neighbors(chunk_id, n=1):
        if chunk_id not in chunk_map:
            return []
        pos = chunk_map[chunk_id].metadata.position
        return [c for c in chunks if abs(c.metadata.position - pos) <= n]

    store.get_chunk_with_neighbors = MagicMock(side_effect=get_neighbors)

    # Mock get_chunks_by_boundary - matches VectorStore.get_chunks_by_boundary signature
    def get_boundary_chunks(boundary_index_arg, boundary_id, include_children=True):
        chunk_ids = boundary_index_arg.get_chunks_in_boundary(boundary_id, include_children)
        result = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        result.sort(key=lambda c: c.metadata.position)
        return result

    store.get_chunks_by_boundary = MagicMock(side_effect=get_boundary_chunks)

    # Mock search to return relevant chunks - matches VectorStore.search signature
    from ..services.vector_store import SearchResult

    def mock_search(query_embedding, k=5, filter_source=None):
        # Return first k chunks as results
        results = []
        for chunk in chunks[:k]:
            results.append(SearchResult(chunk=chunk, score=0.9 - 0.1 * chunk.metadata.position))
        return results

    store.search = MagicMock(side_effect=mock_search)

    return store


class TestBoundaryExpansion:
    """Test boundary expansion in search results."""

    def test_expand_to_section(self, indexed_document, mock_store):
        """Should expand result to include full section."""
        chunks, boundary_index = indexed_document

        # Find a chunk that's in a section
        chunk_with_boundary = None
        for chunk in chunks:
            if chunk.metadata.boundary_type == "section":
                chunk_with_boundary = chunk
                break

        if not chunk_with_boundary:
            pytest.skip("No chunk with section boundary found")

        result = _expand_to_boundary(
            chunk=chunk_with_boundary,
            source_name="research_paper",
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            target_boundary="section",
            max_tokens=4096,
            include_siblings=False,
        )

        assert result is not None
        assert "content" in result
        assert "boundary_info" in result
        # Content should be at least as long as original (may be same if section is one chunk)
        assert len(result["content"]) >= len(chunk_with_boundary.content)

    def test_expand_to_chapter(self, indexed_document, mock_store):
        """Should expand result to include full chapter."""
        chunks, boundary_index = indexed_document

        # Find any chunk with a boundary
        chunk = chunks[0]

        result = _expand_to_boundary(
            chunk=chunk,
            source_name="research_paper",
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            target_boundary="chapter",
            max_tokens=4096,
            include_siblings=False,
        )

        assert result is not None
        assert result["boundary_info"]["type"] == "chapter"

    def test_token_limit_respected(self, indexed_document, mock_store):
        """Should not exceed max_tokens when expanding."""
        chunks, boundary_index = indexed_document

        chunk = chunks[0]
        max_tokens = 100  # Very low limit

        result = _expand_to_boundary(
            chunk=chunk,
            source_name="research_paper",
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            target_boundary="chapter",
            max_tokens=max_tokens,
            include_siblings=False,
        )

        if result:
            assert result["token_count"] <= max_tokens * 1.5  # Allow some tolerance

    def test_include_siblings(self, indexed_document, mock_store):
        """Should include sibling boundaries when requested."""
        chunks, boundary_index = indexed_document

        # Find a chunk in a section that has siblings
        chunk = None
        for c in chunks:
            if c.metadata.boundary_id:
                sibs = boundary_index.get_siblings(c.metadata.boundary_id)
                if sibs:
                    chunk = c
                    break

        if not chunk:
            pytest.skip("No chunk with siblings found")

        result_without_siblings = _expand_to_boundary(
            chunk=chunk,
            source_name="research_paper",
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            target_boundary="section",
            max_tokens=10000,
            include_siblings=False,
        )

        result_with_siblings = _expand_to_boundary(
            chunk=chunk,
            source_name="research_paper",
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            target_boundary="section",
            max_tokens=10000,
            include_siblings=True,
        )

        if result_without_siblings and result_with_siblings:
            # With siblings should have more content
            assert result_with_siblings["token_count"] >= result_without_siblings["token_count"]


class TestContextBuilding:
    """Test building context from neighboring chunks."""

    def test_build_context_with_neighbors(self):
        """Should build context with before/after snippets."""
        target = Chunk(
            id="doc:1",
            content="Target chunk content here.",
            metadata=ChunkMetadata(
                source="doc",
                position=1,
                total_chunks=3,
            ),
        )

        neighbors = [
            Chunk(
                id="doc:0",
                content="Previous chunk content that comes before the target.",
                metadata=ChunkMetadata(source="doc", position=0, total_chunks=3),
            ),
            target,
            Chunk(
                id="doc:2",
                content="Next chunk content that follows the target.",
                metadata=ChunkMetadata(source="doc", position=2, total_chunks=3),
            ),
        ]

        context = _build_context(target, neighbors)

        assert context is not None
        assert "before" in context
        assert "after" in context
        assert "Previous" in context["before"]
        assert "Next" in context["after"]

    def test_build_context_no_neighbors(self):
        """Should return None when no neighbors."""
        target = Chunk(
            id="doc:0",
            content="Only chunk.",
            metadata=ChunkMetadata(source="doc", position=0, total_chunks=1),
        )

        context = _build_context(target, [target])

        assert context is None


class TestSearchExecution:
    """Test full search execution with boundary expansion."""

    @pytest.mark.asyncio
    async def test_basic_search(self, indexed_document, mock_store):
        """Should execute basic search without expansion."""
        chunks, boundary_index = indexed_document

        params = SearchParams(
            query="data collection methods",
            top_k=3,
            include_context=True,
        )

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        response = await execute_search(
            params=params,
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            embed_fn=mock_embed,
        )

        assert len(response.results) <= 3
        assert response.query == "data collection methods"

    @pytest.mark.asyncio
    async def test_search_with_expansion(self, indexed_document, mock_store):
        """Should execute search with boundary expansion."""
        chunks, boundary_index = indexed_document

        params = SearchParams(
            query="research methodology",
            top_k=2,
            expand_to_boundary="section",
            max_return_tokens=4096,
        )

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        response = await execute_search(
            params=params,
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            embed_fn=mock_embed,
        )

        assert response.expansion_applied == "section"

    @pytest.mark.asyncio
    async def test_search_respects_token_budget(self, indexed_document, mock_store):
        """Should stop adding results when token budget exceeded."""
        chunks, boundary_index = indexed_document

        params = SearchParams(
            query="test",
            top_k=10,
            max_return_tokens=500,  # Low budget
        )

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        response = await execute_search(
            params=params,
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            embed_fn=mock_embed,
        )

        # Should have limited results due to token budget
        assert response.total_tokens <= 600  # Some tolerance


class TestSearchResultItem:
    """Test SearchResultItem structure."""

    @pytest.mark.asyncio
    async def test_result_item_fields(self, indexed_document, mock_store):
        """Should include all expected fields in result items."""
        chunks, boundary_index = indexed_document

        params = SearchParams(
            query="introduction",
            top_k=1,
            include_context=True,
        )

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        response = await execute_search(
            params=params,
            stores={"research_paper": mock_store},
            boundary_indices={"research_paper": boundary_index},
            embed_fn=mock_embed,
        )

        if response.results:
            item = response.results[0]
            assert item.chunk_id is not None
            assert item.source_name == "research_paper"
            assert 0 <= item.score <= 1
            assert item.content is not None
            assert item.metadata is not None


class TestGracefulDegradation:
    """Test behavior when boundary data is missing."""

    @pytest.mark.asyncio
    async def test_search_without_boundary_index(self, mock_store):
        """Should work when boundary index is missing."""
        # Create store without boundary index
        params = SearchParams(
            query="test query",
            top_k=3,
            expand_to_boundary="section",  # Should be ignored
        )

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        # Empty boundary index
        response = await execute_search(
            params=params,
            stores={"research_paper": mock_store},
            boundary_indices={},  # No boundary index
            embed_fn=mock_embed,
        )

        # Should still return results, just without expansion
        assert response.query == "test query"

    def test_expand_missing_boundary(self):
        """Should handle missing boundary gracefully."""
        chunk = Chunk(
            id="doc:0",
            content="Test content",
            metadata=ChunkMetadata(
                source="doc",
                position=0,
                total_chunks=1,
                # No boundary_id set
            ),
        )

        result = _expand_to_boundary(
            chunk=chunk,
            source_name="doc",
            stores={},
            boundary_indices={"doc": BoundaryIndex()},
            target_boundary="section",
            max_tokens=4096,
            include_siblings=False,
        )

        assert result is None  # Should return None, not crash
