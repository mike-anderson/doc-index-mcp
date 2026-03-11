"""Shared fixtures for integration tests."""

import os
import asyncio
import pytest

from src.services.document_loader import load_document
from src.services.chunker import chunk_document, ChunkOptions
from src.services.vector_store import VectorStore
from src.services.embedder import Embedder
from src.tools.search_tool import execute_search, SearchParams

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture_path(name: str) -> str:
    path = os.path.join(FIXTURES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture {name} not found — run download script first")
    return path


# Shared embedder (expensive to init, reuse across tests)
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@pytest.fixture(scope="session")
def embedder():
    return get_embedder()


@pytest.fixture(scope="session")
def nist_csf_indexed(embedder):
    """Index NIST CSF 2.0 once for the session."""
    return _index_doc("nist-csf-2.0.pdf", embedder)


@pytest.fixture(scope="session")
def nist_800_53_indexed(embedder):
    """Index NIST 800-53 Rev 5 once for the session."""
    return _index_doc("nist-800-53r5.pdf", embedder)


@pytest.fixture(scope="session")
def economic_report_indexed(embedder):
    """Index Economic Report of the President 2024 once for the session."""
    return _index_doc("economic-report-2024.pdf", embedder)


def _index_doc(filename, embedder):
    """Load, chunk, embed, and store a document. Returns (store, boundary_index, source_name)."""
    path = _fixture_path(filename)
    loop = asyncio.new_event_loop()
    try:
        doc = loop.run_until_complete(load_document(path))
        source_name = os.path.splitext(filename)[0]

        options = ChunkOptions()
        chunks, boundary_index = chunk_document(
            doc.content, source_name, options,
            loader_boundaries=doc.boundaries or None,
        )

        texts = [c.content for c in chunks]
        embeddings = loop.run_until_complete(embedder.embed_texts(texts))

        store = VectorStore(dimension=embedder.dimension)
        loop.run_until_complete(store.initialize(max_elements=max(10000, len(chunks) * 2)))
        loop.run_until_complete(store.add_chunks(chunks, embeddings))

        return store, boundary_index, source_name
    finally:
        loop.close()


async def run_search(stores, boundary_indices, embedder, **kwargs):
    """Helper to run a search with given params."""
    params = SearchParams(**kwargs)
    return await execute_search(
        params=params,
        stores=stores,
        boundary_indices=boundary_indices,
        embed_fn=embedder.embed_text,
    )
