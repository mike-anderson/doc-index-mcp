"""
Tests for document chunking.

Tests boundary-aware chunking algorithm, token counting, and boundary index functionality.
"""

import pytest
from ..services.chunker import (
    Chunk,
    ChunkMetadata,
    ChunkOptions,
    BoundaryIndex,
    count_tokens,
    chunk_document,
)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_simple_text(self):
        """Should count tokens in simple text."""
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert count <= 3  # Should be 2-3 tokens

    def test_count_empty_string(self):
        """Should return 0 for empty string."""
        assert count_tokens("") == 0

    def test_count_longer_text(self):
        """Should count tokens in longer text."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = count_tokens(text)
        assert count > 50  # Should be substantial


class TestChunkOptions:
    """Test ChunkOptions validation."""

    def test_valid_options(self):
        """Should not raise for valid options."""
        options = ChunkOptions(
            chunk_size=256,
            chunk_overlap=32,
            max_chunk_size=1024,
            min_chunk_size=64,
        )
        options.validate()  # Should not raise

    def test_invalid_chunk_size_too_small(self):
        """Should raise for chunk_size < 32."""
        options = ChunkOptions(chunk_size=16)
        with pytest.raises(ValueError, match="chunk_size must be at least 32"):
            options.validate()

    def test_invalid_overlap_too_large(self):
        """Should raise for overlap >= chunk_size."""
        options = ChunkOptions(chunk_size=256, chunk_overlap=300)
        with pytest.raises(ValueError, match="chunk_overlap must be less than"):
            options.validate()

    def test_invalid_max_chunk_size(self):
        """Should raise for max_chunk_size < chunk_size."""
        options = ChunkOptions(chunk_size=256, max_chunk_size=128)
        with pytest.raises(ValueError, match="max_chunk_size must be >="):
            options.validate()

    def test_invalid_min_chunk_size(self):
        """Should raise for min_chunk_size > chunk_size."""
        options = ChunkOptions(chunk_size=256, min_chunk_size=512)
        with pytest.raises(ValueError, match="min_chunk_size must be <="):
            options.validate()


class TestChunkDocument:
    """Test boundary-aware chunking algorithm."""

    def test_basic_chunking(self):
        """Should chunk with boundary-aware algorithm."""
        content = """# Chapter 1

Introduction text here. This is the first paragraph.

## Section 1.1

Section content here. More text for the section.
"""
        chunks, boundary_index = chunk_document(content, "test_doc")

        assert len(chunks) >= 1
        assert isinstance(boundary_index, BoundaryIndex)

    def test_respects_boundaries(self):
        """Should track boundary metadata in chunks."""
        content = """# Chapter 1

Chapter content here.

## Section 1.1

Section content here.
"""
        chunks, boundary_index = chunk_document(content, "test_doc")

        # At least some chunks should have boundary info
        chunks_with_boundaries = [c for c in chunks if c.metadata.boundary_id]
        assert len(chunks_with_boundaries) > 0

    def test_creates_boundary_index(self):
        """Should create proper boundary index."""
        content = """# Chapter

## Section A

Content A.

## Section B

Content B.
"""
        chunks, boundary_index = chunk_document(content, "test_doc")

        # Should have boundaries detected
        assert len(boundary_index.boundaries) >= 2

        # Should have chunk-to-boundary mappings
        assert len(boundary_index.chunk_to_boundary) > 0

    def test_target_chunk_size(self):
        """Should create smaller chunks targeting embedding model."""
        # Create content that would be multiple chunks
        content = "This is a sentence. " * 200

        chunks, _ = chunk_document(content, "test_doc")

        # With 256 token target, we should have multiple chunks
        assert len(chunks) > 1

        # Most chunks should be around 256 tokens
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            token_count = chunk.metadata.token_count or count_tokens(chunk.content)
            # Allow some variance but should be reasonable
            assert token_count <= 1024  # Should not exceed max

    def test_never_splits_mid_sentence(self):
        """Should never split in the middle of a sentence."""
        content = "First sentence ends here. Second sentence continues. Third sentence follows. " * 50

        chunks, _ = chunk_document(content, "test_doc")

        for chunk in chunks:
            # Each chunk should end at a sentence boundary or with complete text
            text = chunk.content.strip()
            if text:
                # Should end with sentence punctuation or be at document end
                assert text[-1] in ".!?\"'" or chunk.metadata.position == len(chunks) - 1

    def test_handles_code_blocks(self):
        """Should handle code blocks without splitting them."""
        content = """# Code Example

Here is some code:

```python
def hello():
    print("Hello, World!")
    return True
```

More text after the code.
"""
        chunks, _ = chunk_document(content, "test_doc")

        # Find chunk with code
        code_chunks = [c for c in chunks if "```python" in c.content]
        assert len(code_chunks) >= 1

        # Code block should be complete
        for chunk in code_chunks:
            if "```python" in chunk.content:
                assert "```" in chunk.content.split("```python")[1]

    def test_min_chunk_size(self):
        """Should merge tiny trailing chunks."""
        content = "Short paragraph.\n\nAnother short one."

        options = ChunkOptions(min_chunk_size=64)
        chunks, _ = chunk_document(content, "test_doc", options)

        # Tiny content should be merged, not left as separate chunk
        for chunk in chunks:
            token_count = chunk.metadata.token_count or count_tokens(chunk.content)
            # Either meets min size or is the only chunk
            assert token_count >= options.min_chunk_size or len(chunks) == 1

    def test_chunk_ids(self):
        """Should generate correct chunk IDs."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks, _ = chunk_document(content, "my_doc")

        for i, chunk in enumerate(chunks):
            assert chunk.id == f"my_doc:{i}"

    def test_chunk_metadata(self):
        """Should populate chunk metadata correctly."""
        content = "Content here.\n\n" * 50
        chunks, _ = chunk_document(content, "test")

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.source == "test"
            assert chunk.metadata.position == i
            assert chunk.metadata.total_chunks == len(chunks)


class TestBoundaryIndex:
    """Test BoundaryIndex functionality."""

    def test_add_boundary(self):
        """Should add boundary and track hierarchy."""
        from ..services.chunker import Boundary, BoundaryType

        index = BoundaryIndex()

        chapter = Boundary(
            type=BoundaryType.CHAPTER,
            level=1,
            id="chapter:1",
            title="Chapter 1",
            start_offset=0,
        )
        index.add_boundary(chapter)

        section = Boundary(
            type=BoundaryType.SECTION,
            level=2,
            id="section:1",
            title="Section 1",
            start_offset=100,
            parent_id="chapter:1",
        )
        index.add_boundary(section)

        assert len(index.boundaries) == 2
        assert "chapter:1" in index.hierarchy
        assert "section:1" in index.hierarchy["chapter:1"]

    def test_map_chunk_to_boundary(self):
        """Should map chunks to boundaries bidirectionally."""
        index = BoundaryIndex()

        index.map_chunk_to_boundary("doc:0", "section:1")
        index.map_chunk_to_boundary("doc:1", "section:1")
        index.map_chunk_to_boundary("doc:2", "section:2")

        assert index.chunk_to_boundary["doc:0"] == "section:1"
        assert "doc:0" in index.boundary_to_chunks["section:1"]
        assert "doc:1" in index.boundary_to_chunks["section:1"]
        assert "doc:2" in index.boundary_to_chunks["section:2"]

    def test_get_chunks_in_boundary(self):
        """Should retrieve all chunks in a boundary."""
        index = BoundaryIndex()

        index.map_chunk_to_boundary("doc:0", "section:1")
        index.map_chunk_to_boundary("doc:1", "section:1")

        chunks = index.get_chunks_in_boundary("section:1")
        assert set(chunks) == {"doc:0", "doc:1"}

    def test_get_ancestor_at_level(self):
        """Should find ancestor at specified level."""
        from ..services.chunker import Boundary, BoundaryType

        index = BoundaryIndex()

        chapter = Boundary(
            type=BoundaryType.CHAPTER,
            level=1,
            id="chapter:1",
            title="Chapter",
            start_offset=0,
        )
        section = Boundary(
            type=BoundaryType.SECTION,
            level=2,
            id="section:1",
            title="Section",
            start_offset=50,
            parent_id="chapter:1",
        )
        subsection = Boundary(
            type=BoundaryType.SUBSECTION,
            level=3,
            id="subsection:1",
            title="Subsection",
            start_offset=100,
            parent_id="section:1",
        )

        index.add_boundary(chapter)
        index.add_boundary(section)
        index.add_boundary(subsection)

        # From subsection, should find chapter at level 1
        ancestor = index.get_ancestor_at_level("subsection:1", 1)
        assert ancestor == "chapter:1"

        # From subsection, should find section at level 2
        ancestor = index.get_ancestor_at_level("subsection:1", 2)
        assert ancestor == "section:1"

    def test_get_siblings(self):
        """Should find sibling boundaries."""
        from ..services.chunker import Boundary, BoundaryType

        index = BoundaryIndex()

        chapter = Boundary(
            type=BoundaryType.CHAPTER,
            level=1,
            id="chapter:1",
            title="Chapter",
            start_offset=0,
        )
        section_a = Boundary(
            type=BoundaryType.SECTION,
            level=2,
            id="section:1",
            title="Section A",
            start_offset=50,
            parent_id="chapter:1",
        )
        section_b = Boundary(
            type=BoundaryType.SECTION,
            level=2,
            id="section:2",
            title="Section B",
            start_offset=100,
            parent_id="chapter:1",
        )

        index.add_boundary(chapter)
        index.add_boundary(section_a)
        index.add_boundary(section_b)

        siblings = index.get_siblings("section:1")
        assert siblings == ["section:2"]

    def test_serialization(self, tmp_path):
        """Should serialize and deserialize correctly."""
        from ..services.chunker import Boundary, BoundaryType

        index = BoundaryIndex()
        index.add_boundary(Boundary(
            type=BoundaryType.CHAPTER,
            level=1,
            id="chapter:1",
            title="Test Chapter",
            start_offset=0,
        ))
        index.map_chunk_to_boundary("doc:0", "chapter:1")

        # Save
        path = tmp_path / "boundaries.json"
        index.save(str(path))

        # Load
        loaded = BoundaryIndex.load(str(path))

        assert len(loaded.boundaries) == 1
        assert loaded.boundaries[0].title == "Test Chapter"
        assert loaded.chunk_to_boundary["doc:0"] == "chapter:1"
