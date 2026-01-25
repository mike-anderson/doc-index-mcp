"""Tests for the vector store service."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ..services.chunker import Chunk, ChunkMetadata
from ..services.vector_store import VectorStore


def make_chunk(source: str, position: int, content: str = "test content") -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        id=f"{source}:{position}",
        content=content,
        metadata=ChunkMetadata(
            source=source,
            position=position,
            total_chunks=10,
        ),
    )


def random_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Create random normalized embeddings."""
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test store initialization."""
        store = VectorStore(dimension=384)
        await store.initialize()

        assert store.index is not None
        assert len(store.chunks) == 0

    @pytest.mark.asyncio
    async def test_add_chunks(self):
        """Test adding chunks to store."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(5)]
        embeddings = random_embeddings(5)

        await store.add_chunks(chunks, embeddings)

        assert len(store.chunks) == 5

    @pytest.mark.asyncio
    async def test_add_chunks_initializes_if_needed(self):
        """Test that add_chunks initializes index if not done."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", 0)]
        embeddings = random_embeddings(1)

        await store.add_chunks(chunks, embeddings)

        assert store.index is not None

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Test searching empty store."""
        store = VectorStore(dimension=384)
        await store.initialize()

        query = random_embeddings(1)[0]
        results = store.search(query, k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that search returns results."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i, f"content {i}") for i in range(10)]
        embeddings = random_embeddings(10)

        await store.add_chunks(chunks, embeddings)

        query = embeddings[0]  # Search for first embedding
        results = store.search(query, k=3)

        assert len(results) == 3
        assert all(hasattr(r, 'chunk') and hasattr(r, 'score') for r in results)

    @pytest.mark.asyncio
    async def test_search_finds_exact_match(self):
        """Test that exact match gets highest score."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(5)]
        embeddings = random_embeddings(5)

        await store.add_chunks(chunks, embeddings)

        # Search for exact embedding
        query = embeddings[2]
        results = store.search(query, k=1)

        assert len(results) == 1
        # Should find the matching chunk
        assert results[0].chunk.id == "test:2"
        # Score should be very high (close to 1.0 for normalized vectors)
        assert results[0].score > 0.99

    @pytest.mark.asyncio
    async def test_search_respects_k(self):
        """Test that search respects k parameter."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(20)]
        embeddings = random_embeddings(20)

        await store.add_chunks(chunks, embeddings)

        query = random_embeddings(1)[0]

        for k in [1, 5, 10]:
            results = store.search(query, k=k)
            assert len(results) == k

    @pytest.mark.asyncio
    async def test_search_k_larger_than_store(self):
        """Test search when k > number of chunks."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(3)]
        embeddings = random_embeddings(3)

        await store.add_chunks(chunks, embeddings)

        query = random_embeddings(1)[0]
        results = store.search(query, k=10)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_chunk_by_id(self):
        """Test retrieving chunk by ID."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i, f"content {i}") for i in range(5)]
        embeddings = random_embeddings(5)

        await store.add_chunks(chunks, embeddings)

        chunk = store.get_chunk_by_id("test:2")

        assert chunk is not None
        assert chunk.id == "test:2"
        assert chunk.content == "content 2"

    @pytest.mark.asyncio
    async def test_get_chunk_by_id_not_found(self):
        """Test retrieving non-existent chunk."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(3)]
        embeddings = random_embeddings(3)

        await store.add_chunks(chunks, embeddings)

        chunk = store.get_chunk_by_id("nonexistent:99")

        assert chunk is None

    @pytest.mark.asyncio
    async def test_get_chunk_with_neighbors(self):
        """Test retrieving chunk with neighbors."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(10)]
        embeddings = random_embeddings(10)

        await store.add_chunks(chunks, embeddings)

        neighbors = store.get_chunk_with_neighbors("test:5", n=2)

        assert len(neighbors) == 5  # 2 before + target + 2 after
        ids = [c.id for c in neighbors]
        assert ids == ["test:3", "test:4", "test:5", "test:6", "test:7"]

    @pytest.mark.asyncio
    async def test_get_chunk_with_neighbors_at_start(self):
        """Test neighbors at start of document."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(10)]
        embeddings = random_embeddings(10)

        await store.add_chunks(chunks, embeddings)

        neighbors = store.get_chunk_with_neighbors("test:1", n=3)

        # Should only have 1 before (at position 0)
        ids = [c.id for c in neighbors]
        assert "test:0" in ids
        assert "test:1" in ids

    @pytest.mark.asyncio
    async def test_get_chunk_with_neighbors_at_end(self):
        """Test neighbors at end of document."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(10)]
        embeddings = random_embeddings(10)

        await store.add_chunks(chunks, embeddings)

        neighbors = store.get_chunk_with_neighbors("test:8", n=3)

        # Should only have 1 after (at position 9)
        ids = [c.id for c in neighbors]
        assert "test:8" in ids
        assert "test:9" in ids

    @pytest.mark.asyncio
    async def test_get_chunk_with_neighbors_not_found(self):
        """Test neighbors for non-existent chunk."""
        store = VectorStore(dimension=384)

        chunks = [make_chunk("test", i) for i in range(5)]
        embeddings = random_embeddings(5)

        await store.add_chunks(chunks, embeddings)

        neighbors = store.get_chunk_with_neighbors("nonexistent:99", n=2)

        assert neighbors == []

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save store
            store1 = VectorStore(dimension=384)

            chunks = [make_chunk("test", i, f"content {i}") for i in range(5)]
            embeddings = random_embeddings(5)

            await store1.add_chunks(chunks, embeddings)
            await store1.save(tmpdir)

            # Verify files exist (now uses usearch and jsonl)
            assert (Path(tmpdir) / "index.usearch").exists()
            assert (Path(tmpdir) / "chunks.jsonl").exists()

            # Load into new store
            store2 = VectorStore(dimension=384)
            await store2.load(tmpdir)

            # Verify loaded data
            assert len(store2.chunks) == 5
            assert store2.get_chunk_by_id("test:2").content == "content 2"

            # Verify search still works
            query = embeddings[0]
            results = store2.search(query, k=1)
            assert results[0].chunk.id == "test:0"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        """Test loading from non-existent path."""
        store = VectorStore(dimension=384)

        with pytest.raises(FileNotFoundError):
            await store.load("/nonexistent/path")

    @pytest.mark.asyncio
    async def test_multiple_sources(self):
        """Test store with chunks from multiple sources."""
        store = VectorStore(dimension=384)

        chunks = [
            make_chunk("book1", 0, "book1 content 0"),
            make_chunk("book1", 1, "book1 content 1"),
            make_chunk("book2", 0, "book2 content 0"),
            make_chunk("book2", 1, "book2 content 1"),
        ]
        embeddings = random_embeddings(4)

        await store.add_chunks(chunks, embeddings)

        # Should find chunks from both sources
        assert store.get_chunk_by_id("book1:0") is not None
        assert store.get_chunk_by_id("book2:1") is not None

    @pytest.mark.asyncio
    async def test_neighbors_same_source_only(self):
        """Test that neighbors only returns chunks from same source."""
        store = VectorStore(dimension=384)

        chunks = [
            make_chunk("book1", 0),
            make_chunk("book1", 1),
            make_chunk("book2", 0),  # Different source
            make_chunk("book1", 2),
        ]
        embeddings = random_embeddings(4)

        await store.add_chunks(chunks, embeddings)

        neighbors = store.get_chunk_with_neighbors("book1:1", n=2)

        # Should only include book1 chunks
        sources = {c.metadata.source for c in neighbors}
        assert sources == {"book1"}
