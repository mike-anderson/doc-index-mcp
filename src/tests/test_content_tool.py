"""Tests for the content retrieval tool."""

import pytest
import numpy as np
from ..services.chunker import (
    Boundary, BoundaryType, BoundaryIndex, Chunk, ChunkMetadata,
)
from ..services.vector_store import VectorStore
from ..tools.content_tool import (
    get_content_by_boundary,
    get_content_by_page_range,
    get_content_by_title,
    get_content_by_position_range,
)


@pytest.fixture
def sample_store_and_index():
    """Create a VectorStore and BoundaryIndex with sample data."""
    bi = BoundaryIndex()

    # Chapters and sections
    bi.add_boundary(Boundary(
        type=BoundaryType.CHAPTER, level=1, id="chapter:1",
        title="Introduction", start_offset=0,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.SECTION, level=2, id="section:1",
        title="Background", start_offset=100, parent_id="chapter:1",
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.CHAPTER, level=1, id="chapter:2",
        title="Methods", start_offset=1000,
    ))

    # Pages
    bi.add_boundary(Boundary(
        type=BoundaryType.PAGE, level=4, id="page:1",
        title="1", start_offset=0,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.PAGE, level=4, id="page:2",
        title="2", start_offset=500,
    ))
    bi.add_boundary(Boundary(
        type=BoundaryType.PAGE, level=4, id="page:3",
        title="3", start_offset=1000,
    ))

    # Create chunks
    chunks = []
    for i in range(10):
        page = 1 if i < 3 else (2 if i < 6 else 3)
        chunks.append(Chunk(
            id=f"doc:{i}",
            content=f"Content of chunk {i}. This is test content for testing purposes.",
            metadata=ChunkMetadata(
                source="doc",
                position=i,
                total_chunks=10,
                page=page,
                token_count=15,
            ),
        ))

    # Map chunks to boundaries
    for i in range(3):
        bi.map_chunk_to_boundary(f"doc:{i}", "chapter:1")
    for i in range(3, 6):
        bi.map_chunk_to_boundary(f"doc:{i}", "section:1")
    for i in range(6, 10):
        bi.map_chunk_to_boundary(f"doc:{i}", "chapter:2")

    # Create store (no embeddings needed for content retrieval)
    store = VectorStore(dimension=384)
    store.chunks = chunks
    store._id_to_position = {c.id: i for i, c in enumerate(chunks)}

    return store, bi


class TestGetContentByBoundary:
    def test_retrieves_chapter_with_children(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_boundary(store, bi, "chapter:1")
        # Default include_children=True: 3 direct + 3 from section:1 = 6
        assert resp.chunk_count == 6
        assert "chunk 0" in resp.content
        assert resp.boundary_info["title"] == "Introduction"

    def test_includes_children_explicit(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_boundary(store, bi, "chapter:1", include_children=True)
        assert resp.chunk_count == 6

    def test_excludes_children(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_boundary(store, bi, "chapter:1", include_children=False)
        assert resp.chunk_count == 3

    def test_respects_token_budget(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_boundary(store, bi, "chapter:2", max_tokens=30)
        assert resp.total_tokens <= 30
        assert resp.truncated is True

    def test_nonexistent_boundary(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_boundary(store, bi, "chapter:999")
        assert resp.content == ""
        assert resp.chunk_count == 0


class TestGetContentByPageRange:
    def test_single_page(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_page_range(store, bi, 1, 1)
        assert resp.chunk_count == 3
        assert "chunk 0" in resp.content

    def test_page_range(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_page_range(store, bi, 1, 2)
        assert resp.chunk_count == 6

    def test_all_pages(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_page_range(store, bi, 1, 3)
        assert resp.chunk_count == 10

    def test_nonexistent_page(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_page_range(store, bi, 99, 100)
        assert resp.chunk_count == 0
        assert resp.content == ""


class TestGetContentByTitle:
    def test_exact_title(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_title(store, bi, "Introduction")
        assert resp.chunk_count > 0
        assert resp.boundary_info["title"] == "Introduction"

    def test_fuzzy_title(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_title(store, bi, "Intro")
        assert resp.chunk_count > 0

    def test_type_filter(self, sample_store_and_index):
        store, bi = sample_store_and_index
        # "Survey Design" doesn't exist, so with chapter filter nothing should match well
        resp = get_content_by_title(store, bi, "xyznonexistent", boundary_type="chapter")
        assert resp.chunk_count == 0

    def test_no_match(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_title(store, bi, "zzzzzzzzzzz")
        assert resp.chunk_count == 0


class TestGetContentByPositionRange:
    def test_position_range(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_position_range(store, 0, 2)
        assert resp.chunk_count == 3

    def test_single_position(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_position_range(store, 5, 5)
        assert resp.chunk_count == 1
        assert "chunk 5" in resp.content

    def test_respects_token_budget(self, sample_store_and_index):
        store, bi = sample_store_and_index
        resp = get_content_by_position_range(store, 0, 9, max_tokens=30)
        assert resp.truncated is True
        assert resp.total_tokens <= 30
