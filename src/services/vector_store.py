"""
Vector store service for semantic search using USearch index.

Provides efficient similarity search over document chunk embeddings
with support for boundary-aware retrieval. Uses USearch for HNSW-based
search with pre-built wheels (no C++ compilation required).
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from usearch.index import Index

from .chunker import Chunk, BoundaryIndex


@dataclass
class SearchResult:
    """A single search result with chunk and similarity score."""
    chunk: Chunk
    score: float  # Similarity score (0-1, higher is better)


class VectorStore:
    """
    Vector store using USearch (HNSW-based index with pre-built wheels).

    Provides:
    - Fast approximate nearest neighbor search
    - Persistence to disk
    - Integration with boundary indices
    - No C++ compilation required (pre-built wheels)
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.

        Args:
            dimension: Embedding vector dimension (384 for bge-small-en-v1.5)
        """
        self.dimension = dimension
        self.index: Optional[Index] = None
        self.chunks: list[Chunk] = []
        self._id_to_position: dict[str, int] = {}

    async def initialize(self, max_elements: int = 10000):
        """
        Initialize the USearch index.

        Args:
            max_elements: Maximum number of elements the index can hold
        """
        self.index = Index(
            ndim=self.dimension,
            metric='cos',  # Cosine similarity
            dtype='f32',   # float32 vectors
        )

    async def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray):
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: Numpy array of shape (n_chunks, dimension)
        """
        if self.index is None:
            await self.initialize(max_elements=max(10000, len(chunks) * 2))

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")

        # Assign positions and add to index
        start_pos = len(self.chunks)
        positions = np.arange(start_pos, start_pos + len(chunks), dtype=np.uint64)

        # USearch expects (keys, vectors) - keys are uint64 IDs
        self.index.add(positions, embeddings.astype(np.float32))

        # Store chunks and build lookup
        for pos, chunk in zip(positions, chunks):
            self.chunks.append(chunk)
            self._id_to_position[chunk.id] = int(pos)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_source: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_source: Optional source name to filter results

        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        # Fetch more results if filtering, to ensure we get enough after filter
        fetch_k = k * 3 if filter_source else k

        # USearch search - pass 1D vector directly
        matches = self.index.search(
            query_embedding.astype(np.float32),
            min(fetch_k, len(self.chunks))
        )

        results = []

        # Handle both single and batch results from usearch
        # matches is a Matches object with keys and distances attributes
        keys = np.atleast_1d(matches.keys)
        distances = np.atleast_1d(matches.distances)

        for label, distance in zip(keys, distances):
            label = int(label)
            if label >= len(self.chunks):
                continue

            chunk = self.chunks[label]

            # Apply source filter
            if filter_source and chunk.metadata.source != filter_source:
                continue

            # USearch cosine metric returns distance (0=identical, 2=opposite)
            # Convert to similarity score (1 - distance/2) for 0-1 range
            score = 1.0 - float(distance) / 2.0
            results.append(SearchResult(chunk=chunk, score=score))

            if len(results) >= k:
                break

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        pos = self._id_to_position.get(chunk_id)
        if pos is not None and pos < len(self.chunks):
            return self.chunks[pos]
        return None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get multiple chunks by their IDs, preserving order."""
        result = []
        for cid in chunk_ids:
            chunk = self.get_chunk_by_id(cid)
            if chunk:
                result.append(chunk)
        return result

    def get_chunk_with_neighbors(self, chunk_id: str, n: int = 1) -> list[Chunk]:
        """
        Get a chunk along with its n neighbors on each side.

        Args:
            chunk_id: ID of the target chunk
            n: Number of neighbors on each side

        Returns:
            List of chunks [prev_n, ..., prev_1, target, next_1, ..., next_n]
        """
        pos = self._id_to_position.get(chunk_id)
        if pos is None:
            return []

        # Get source to only include neighbors from same source
        target_chunk = self.chunks[pos]
        source = target_chunk.metadata.source

        # Find neighbors from same source
        start = max(0, pos - n)
        end = min(len(self.chunks), pos + n + 1)

        result = []
        for i in range(start, end):
            chunk = self.chunks[i]
            if chunk.metadata.source == source:
                result.append(chunk)

        return result

    def get_chunks_by_boundary(
        self,
        boundary_index: BoundaryIndex,
        boundary_id: str,
        include_children: bool = True
    ) -> list[Chunk]:
        """
        Get all chunks within a boundary.

        Args:
            boundary_index: The boundary index to use
            boundary_id: ID of the boundary
            include_children: Whether to include chunks from child boundaries

        Returns:
            List of chunks within the boundary, in order
        """
        chunk_ids = boundary_index.get_chunks_in_boundary(boundary_id, include_children)

        # Get chunks and sort by position
        chunks = self.get_chunks_by_ids(chunk_ids)
        chunks.sort(key=lambda c: c.metadata.position)

        return chunks

    async def save(self, base_path: str):
        """
        Save the vector store to disk.

        Creates:
        - {base_path}/index.usearch - USearch index binary
        - {base_path}/chunks.jsonl - Chunk metadata (JSON Lines format)

        Args:
            base_path: Directory to save files
        """
        os.makedirs(base_path, exist_ok=True)

        # Save USearch index
        if self.index is not None and len(self.chunks) > 0:
            index_path = os.path.join(base_path, "index.usearch")
            self.index.save(index_path)

        # Save chunks as JSONL (one JSON object per line for streaming access)
        chunks_path = os.path.join(base_path, "chunks.jsonl")
        with open(chunks_path, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

    async def load(self, base_path: str):
        """
        Load the vector store from disk.

        Args:
            base_path: Directory containing saved files
        """
        # Load chunks - try JSONL first, fall back to JSON for backwards compatibility
        chunks_path_jsonl = os.path.join(base_path, "chunks.jsonl")
        chunks_path_json = os.path.join(base_path, "chunks.json")

        if os.path.exists(chunks_path_jsonl):
            # Load from JSONL (streaming format)
            self.chunks = []
            with open(chunks_path_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.chunks.append(Chunk.from_dict(json.loads(line)))
        elif os.path.exists(chunks_path_json):
            # Backwards compatibility: load from JSON
            with open(chunks_path_json, "r") as f:
                chunks_data = json.load(f)
            self.chunks = [Chunk.from_dict(d) for d in chunks_data]
        else:
            raise FileNotFoundError(f"Chunks file not found: {chunks_path_jsonl}")

        self._id_to_position = {chunk.id: i for i, chunk in enumerate(self.chunks)}

        # Load USearch index
        index_path = os.path.join(base_path, "index.usearch")
        if os.path.exists(index_path) and len(self.chunks) > 0:
            self.index = Index.restore(index_path)

    def get_all_sources(self) -> set[str]:
        """Get all unique source names in the store."""
        return {chunk.metadata.source for chunk in self.chunks}

    def get_chunk_count(self, source: Optional[str] = None) -> int:
        """Get count of chunks, optionally filtered by source."""
        if source is None:
            return len(self.chunks)
        return sum(1 for c in self.chunks if c.metadata.source == source)

    def clear(self):
        """Clear all data from the store."""
        self.index = None
        self.chunks = []
        self._id_to_position = {}
