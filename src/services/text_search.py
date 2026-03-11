"""
Text search service for fuzzy and exact matching against chunks.

Performs O(n) scan over chunk contents using stdlib only (difflib, re).
Designed for local single-file use cases where building a separate
text index is unnecessary.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .chunker import Chunk


# Common English stopwords to skip during fuzzy token matching
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "not", "no", "so", "if", "then", "than",
    "too", "very", "just", "about", "up", "out", "into", "over", "after",
})


@dataclass
class TextMatch:
    """Result of text matching against a single chunk."""
    chunk: Chunk
    score: float          # 0.0 - 1.0
    match_type: str       # "exact_substring" or "fuzzy_token"


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r'\w+', text.lower())


def text_search(
    query: str,
    chunks: list[Chunk],
    min_score: float = 0.3,
) -> list[TextMatch]:
    """
    Perform exact substring and fuzzy token matching against chunks.

    Args:
        query: The search query string
        chunks: List of chunks to search through
        min_score: Minimum score threshold to include in results

    Returns:
        List of TextMatch results sorted by score descending,
        filtered to those above min_score.
    """
    query_lower = query.strip().lower()
    if not query_lower:
        return []

    query_tokens = _tokenize(query)
    # Filter stopwords for fuzzy matching (but keep them for exact substring)
    query_tokens_filtered = [t for t in query_tokens if t not in _STOPWORDS]
    # Fall back to all tokens if everything was a stopword
    if not query_tokens_filtered:
        query_tokens_filtered = query_tokens

    results: list[TextMatch] = []

    for chunk in chunks:
        content_lower = chunk.content.lower()

        # Tier 1: Exact substring match (score 0.85 - 1.0)
        exact_score = _exact_substring_score(query_lower, content_lower)

        # Tier 2: Fuzzy token match (score 0.0 - 0.75)
        fuzzy_score = _fuzzy_token_score(query_tokens_filtered, content_lower)

        # Take the best tier
        if exact_score >= fuzzy_score:
            score = exact_score
            match_type = "exact_substring"
        else:
            score = fuzzy_score
            match_type = "fuzzy_token"

        if score >= min_score:
            results.append(TextMatch(
                chunk=chunk,
                score=score,
                match_type=match_type,
            ))

    results.sort(key=lambda m: -m.score)
    return results


def _exact_substring_score(query_lower: str, content_lower: str) -> float:
    """
    Score based on exact case-insensitive substring match.

    Returns 0.85 - 1.0 based on how much of the chunk the query covers.
    A longer query match relative to chunk length scores higher.
    Returns 0.0 if no match.
    """
    if query_lower not in content_lower:
        return 0.0

    # Scale: 0.85 base + up to 0.15 based on query/chunk length ratio
    coverage = len(query_lower) / max(len(content_lower), 1)
    return 0.85 + 0.15 * min(coverage, 1.0)


def _fuzzy_token_score(
    query_tokens: list[str],
    content_lower: str,
) -> float:
    """
    Score based on fuzzy token-level matching.

    For each query token, finds the best matching token in the chunk
    using SequenceMatcher. A token counts as matched if ratio >= 0.8.

    Returns 0.0 - 0.75 based on fraction of query tokens matched.
    """
    if not query_tokens:
        return 0.0

    content_tokens = _tokenize(content_lower)
    if not content_tokens:
        return 0.0

    # Build a set of unique content tokens for faster matching
    unique_content_tokens = set(content_tokens)

    matched = 0
    for qt in query_tokens:
        # Fast path: exact token match
        if qt in unique_content_tokens:
            matched += 1
            continue

        # Slow path: fuzzy match against unique content tokens
        best_ratio = 0.0
        for ct in unique_content_tokens:
            # Skip tokens with very different lengths (optimization)
            if abs(len(qt) - len(ct)) > max(len(qt), len(ct)) * 0.4:
                continue
            ratio = SequenceMatcher(None, qt, ct).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if ratio >= 0.8:
                    break  # Good enough, stop searching

        if best_ratio >= 0.8:
            matched += 1

    token_match_fraction = matched / len(query_tokens)
    return 0.75 * token_match_fraction
