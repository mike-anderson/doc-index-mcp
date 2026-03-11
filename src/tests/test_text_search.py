"""Tests for the text search service."""

import pytest

from ..services.chunker import Chunk, ChunkMetadata
from ..services.text_search import text_search, _exact_substring_score, _fuzzy_token_score, _tokenize


def make_chunk(position: int, content: str) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        id=f"test:{position}",
        content=content,
        metadata=ChunkMetadata(
            source="test",
            position=position,
            total_chunks=10,
        ),
    )


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        assert _tokenize("Q3 revenue: $4.2M!") == ["q3", "revenue", "4", "2m"]

    def test_empty(self):
        assert _tokenize("") == []


class TestExactSubstringScore:
    def test_exact_match(self):
        score = _exact_substring_score("revenue was $4.2m", "the quarterly revenue was $4.2m in q3")
        assert score >= 0.85
        assert score <= 1.0

    def test_no_match(self):
        assert _exact_substring_score("xyz123", "the quarterly revenue") == 0.0

    def test_full_coverage(self):
        """Query that covers the entire content scores near 1.0."""
        score = _exact_substring_score("hello world", "hello world")
        assert score > 0.99

    def test_partial_coverage(self):
        """Short query in long content scores near 0.85."""
        score = _exact_substring_score("hi", "hi " + "x " * 500)
        assert 0.85 <= score < 0.86


class TestFuzzyTokenScore:
    def test_all_tokens_match(self):
        tokens = ["revenue", "growth"]
        score = _fuzzy_token_score(tokens, "the company saw revenue growth in q3")
        assert score == 0.75

    def test_no_tokens_match(self):
        tokens = ["xyz", "abc"]
        score = _fuzzy_token_score(tokens, "the quarterly revenue was strong")
        assert score == 0.0

    def test_partial_match(self):
        tokens = ["revenue", "xyz"]
        score = _fuzzy_token_score(tokens, "the quarterly revenue was strong")
        assert 0.3 <= score <= 0.4  # 1/2 tokens matched * 0.75

    def test_fuzzy_match(self):
        """Typo in query token should still match via SequenceMatcher."""
        tokens = ["revnue"]  # typo for "revenue"
        score = _fuzzy_token_score(tokens, "the quarterly revenue was strong")
        assert score >= 0.6  # Should fuzzy-match

    def test_empty_query(self):
        assert _fuzzy_token_score([], "some content") == 0.0

    def test_empty_content(self):
        assert _fuzzy_token_score(["hello"], "") == 0.0


class TestTextSearch:
    def test_exact_match_ranks_highest(self):
        chunks = [
            make_chunk(0, "The Q3 revenue was $4.2M, up from Q2."),
            make_chunk(1, "Annual performance review guidelines."),
            make_chunk(2, "Revenue projections for next fiscal year."),
        ]
        results = text_search("Q3 revenue was $4.2M", chunks)
        assert len(results) > 0
        assert results[0].chunk.id == "test:0"
        assert results[0].match_type == "exact_substring"
        assert results[0].score >= 0.85

    def test_no_results_below_threshold(self):
        chunks = [
            make_chunk(0, "Completely unrelated content about gardening tips."),
        ]
        results = text_search("quantum computing algorithms", chunks)
        assert len(results) == 0

    def test_fuzzy_matches_returned(self):
        chunks = [
            make_chunk(0, "The company reported strong revenue growth."),
            make_chunk(1, "Weather forecast for tomorrow is sunny."),
        ]
        results = text_search("revenue growth report", chunks)
        assert len(results) >= 1
        assert results[0].chunk.id == "test:0"

    def test_conceptual_query_low_text_score(self):
        """Conceptual questions with little keyword overlap should score low."""
        chunks = [
            make_chunk(0, "The Q3 earnings were $4.2M with 15% margin improvement."),
            make_chunk(1, "Employee onboarding process takes approximately two weeks."),
        ]
        results = text_search("What are the benefits of the new strategy?", chunks, min_score=0.5)
        # Should have few or no results since it's conceptual
        assert len(results) == 0

    def test_empty_query(self):
        chunks = [make_chunk(0, "Some content")]
        results = text_search("", chunks)
        assert len(results) == 0

    def test_case_insensitive(self):
        chunks = [make_chunk(0, "The QUARTERLY REVENUE was strong.")]
        results = text_search("quarterly revenue", chunks)
        assert len(results) == 1
        assert results[0].match_type == "exact_substring"

    def test_custom_min_score(self):
        chunks = [
            make_chunk(0, "Revenue growth was steady throughout the year."),
        ]
        # High threshold should filter out moderate matches
        results_high = text_search("revenue", chunks, min_score=0.9)
        results_low = text_search("revenue", chunks, min_score=0.1)
        assert len(results_low) >= len(results_high)
