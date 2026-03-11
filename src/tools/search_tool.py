"""
Search tool for the mcp-knowledge MCP server.

Provides hybrid semantic + text search across indexed documents with
boundary-aware expansion capabilities. Semantic search uses vector
similarity, while text search does O(n) exact substring and fuzzy
token matching. Results are merged with sparsity-aware scoring.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    # Relative imports for when running as part of the package (e.g. pytest)
    from ..services.chunker import Chunk, BoundaryIndex, count_tokens
    from ..services.vector_store import VectorStore, SearchResult
    from ..services.text_search import text_search, TextMatch
    from ..services.boundary_detector import get_level_for_boundary_type
except ImportError:
    # Absolute imports for when running with src/ on sys.path (e.g. MCP server)
    from services.chunker import Chunk, BoundaryIndex, count_tokens
    from services.vector_store import VectorStore, SearchResult
    from services.text_search import text_search, TextMatch
    from services.boundary_detector import get_level_for_boundary_type


@dataclass
class SearchParams:
    """Parameters for knowledge search."""
    # Core search parameters
    query: str
    sources: Optional[list[str]] = None
    top_k: int = 5
    include_context: bool = True

    # Boundary expansion parameters
    expand_to_boundary: Optional[str] = None  # 'chapter', 'section', 'subsection', 'page'
    max_return_tokens: int = 4096
    include_siblings: bool = False


@dataclass
class SearchResultItem:
    """A single item in search results."""
    chunk_id: str
    source_name: str
    score: float
    content: str
    metadata: dict
    context: Optional[dict] = None  # {"before": str, "after": str}
    expanded_content: Optional[str] = None  # Full boundary content when expanded
    boundary_info: Optional[dict] = None  # Boundary metadata when expanded
    text_score: Optional[float] = None  # Text match score if applicable
    match_type: Optional[str] = None  # "exact_substring", "fuzzy_token", or None


@dataclass
class SearchResponse:
    """Response from knowledge search."""
    results: list[SearchResultItem] = field(default_factory=list)
    total_tokens: int = 0
    query: str = ""
    expansion_applied: Optional[str] = None


async def execute_search(
    params: SearchParams,
    stores: dict[str, VectorStore],
    boundary_indices: dict[str, BoundaryIndex],
    embed_fn,
) -> SearchResponse:
    """
    Execute a hybrid semantic + text search with optional boundary expansion.

    Runs both vector similarity search and O(n) text matching, then merges
    results using sparsity-aware scoring: few high-confidence text matches
    get boosted, but semantic alignment is always respected.

    Args:
        params: Search parameters
        stores: Map of source_name -> VectorStore
        boundary_indices: Map of source_name -> BoundaryIndex
        embed_fn: Function to embed query text -> numpy array

    Returns:
        SearchResponse with results and metadata
    """
    # Embed the query
    query_embedding = await embed_fn(params.query)

    # --- Semantic search ---
    semantic_by_id: dict[str, tuple[float, Chunk, str]] = {}  # chunk_id -> (score, chunk, source)

    for source_name, store in stores.items():
        if params.sources and source_name not in params.sources:
            continue

        results = store.search(
            query_embedding=query_embedding,
            k=params.top_k * 2,
        )

        for result in results:
            semantic_by_id[result.chunk.id] = (result.score, result.chunk, source_name)

    # --- Text search (O(n) scan) ---
    text_by_id: dict[str, tuple[float, str, Chunk, str]] = {}  # chunk_id -> (score, match_type, chunk, source)

    for source_name, store in stores.items():
        if params.sources and source_name not in params.sources:
            continue

        text_matches = text_search(params.query, store.chunks)
        for match in text_matches:
            text_by_id[match.chunk.id] = (match.score, match.match_type, match.chunk, source_name)

    # --- Merge with sparsity-aware scoring ---
    merged = _merge_results(semantic_by_id, text_by_id, params.top_k)

    # Build response items
    response = SearchResponse(query=params.query)
    total_tokens = 0

    for chunk_id, final_score, chunk, source_name, text_score, match_type in merged:
        item = SearchResultItem(
            chunk_id=chunk.id,
            source_name=source_name,
            score=final_score,
            content=chunk.content,
            metadata={
                "position": chunk.metadata.position,
                "total_chunks": chunk.metadata.total_chunks,
                "boundary_type": chunk.metadata.boundary_type,
                "boundary_id": chunk.metadata.boundary_id,
                "boundary_title": chunk.metadata.boundary_title,
                "token_count": chunk.metadata.token_count,
            },
            text_score=text_score,
            match_type=match_type,
        )

        # Add context (neighboring chunks) if requested
        if params.include_context:
            store = stores[source_name]
            neighbors = store.get_chunk_with_neighbors(chunk.id, n=1)
            context = _build_context(chunk, neighbors)
            if context:
                item.context = context

        # Apply boundary expansion if requested
        if params.expand_to_boundary:
            expanded = _expand_to_boundary(
                chunk=chunk,
                source_name=source_name,
                stores=stores,
                boundary_indices=boundary_indices,
                target_boundary=params.expand_to_boundary,
                max_tokens=params.max_return_tokens - total_tokens,
                include_siblings=params.include_siblings,
            )
            if expanded:
                item.expanded_content = expanded["content"]
                item.boundary_info = expanded["boundary_info"]
                total_tokens += expanded["token_count"]
                response.expansion_applied = params.expand_to_boundary
        else:
            total_tokens += chunk.metadata.token_count or count_tokens(chunk.content)

        response.results.append(item)

        # Stop if we've exceeded token budget
        if total_tokens >= params.max_return_tokens:
            break

    response.total_tokens = total_tokens
    return response


def _merge_results(
    semantic_by_id: dict[str, tuple[float, "Chunk", str]],
    text_by_id: dict[str, tuple[float, str, "Chunk", str]],
    top_k: int,
) -> list[tuple[str, float, "Chunk", str, Optional[float], Optional[str]]]:
    """
    Merge semantic and text search results with sparsity-aware scoring.

    When there are few high-confidence text matches, they get a strong boost.
    When text matches are numerous (common words), they get minimal weight.
    Semantic alignment always contributes significantly.

    Returns:
        List of (chunk_id, final_score, chunk, source_name, text_score, match_type)
        sorted by final_score descending, limited to top_k.
    """
    # Determine text search sparsity and confidence
    text_scores_above_threshold = [
        score for score, _, _, _ in text_by_id.values() if score > 0.3
    ]
    text_hit_count = len(text_scores_above_threshold)

    # Compute text_weight based on sparsity
    if text_hit_count == 0:
        text_weight = 0.0
    elif text_hit_count <= 3:
        avg_confidence = sum(text_scores_above_threshold) / text_hit_count
        text_weight = 0.5 * avg_confidence
    elif text_hit_count <= 10:
        avg_confidence = sum(text_scores_above_threshold) / text_hit_count
        text_weight = 0.3 * avg_confidence
    else:
        text_weight = 0.1

    # Collect all chunk IDs from both result sets
    all_chunk_ids = set(semantic_by_id.keys()) | set(text_by_id.keys())

    scored: list[tuple[str, float, "Chunk", str, Optional[float], Optional[str]]] = []

    for chunk_id in all_chunk_ids:
        sem_entry = semantic_by_id.get(chunk_id)
        txt_entry = text_by_id.get(chunk_id)

        semantic_score = sem_entry[0] if sem_entry else 0.0
        text_score = txt_entry[0] if txt_entry else 0.0
        match_type = txt_entry[1] if txt_entry else None

        # Pick chunk and source from whichever result set has it
        if sem_entry:
            chunk, source_name = sem_entry[1], sem_entry[2]
        else:
            chunk, source_name = txt_entry[2], txt_entry[3]

        # Compute blended score
        final_score = (1 - text_weight) * semantic_score + text_weight * text_score

        # Override: exact substring matches with few hits get a score floor
        # This ensures verbatim quotes always surface near the top
        if text_score >= 0.85 and text_hit_count <= 3:
            final_score = max(final_score, 0.8)

        scored.append((
            chunk_id,
            final_score,
            chunk,
            source_name,
            text_score if text_score > 0 else None,
            match_type,
        ))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def _build_context(target_chunk: Chunk, neighbors: list[Chunk]) -> Optional[dict]:
    """Build context dict with before/after content summaries."""
    if not neighbors:
        return None

    context = {}
    target_pos = target_chunk.metadata.position

    for chunk in neighbors:
        pos = chunk.metadata.position
        if pos < target_pos:
            # Summarize: first 100 chars
            context["before"] = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        elif pos > target_pos:
            context["after"] = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content

    return context if context else None


def _expand_to_boundary(
    chunk: Chunk,
    source_name: str,
    stores: dict[str, VectorStore],
    boundary_indices: dict[str, BoundaryIndex],
    target_boundary: str,
    max_tokens: int,
    include_siblings: bool,
) -> Optional[dict]:
    """
    Expand search result to include full boundary content.

    Args:
        chunk: The matched chunk
        source_name: Source document name
        stores: Vector stores by source
        boundary_indices: Boundary indices by source
        target_boundary: Target boundary type ('chapter', 'section', etc.)
        max_tokens: Maximum tokens to return
        include_siblings: Whether to include sibling boundaries

    Returns:
        Dict with expanded content, boundary info, and token count
    """
    # Get boundary index for this source
    boundary_index = boundary_indices.get(source_name)
    if not boundary_index:
        return None

    # Find the boundary containing this chunk
    chunk_boundary_id = boundary_index.chunk_to_boundary.get(chunk.id)
    if not chunk_boundary_id:
        return None

    # Find ancestor at target level
    target_level = get_level_for_boundary_type(target_boundary)
    target_boundary_id = boundary_index.get_ancestor_at_level(chunk_boundary_id, target_level)

    if not target_boundary_id:
        # Can't expand to requested level, use current boundary
        target_boundary_id = chunk_boundary_id

    # Get all chunks in this boundary
    store = stores[source_name]
    boundary_chunks = store.get_chunks_by_boundary(
        boundary_index,
        target_boundary_id,
        include_children=True
    )

    # Build expanded content respecting token limit
    content_parts = []
    total_tokens = 0

    for boundary_chunk in boundary_chunks:
        chunk_tokens = boundary_chunk.metadata.token_count or count_tokens(boundary_chunk.content)

        if total_tokens + chunk_tokens > max_tokens and content_parts:
            # Would exceed limit, stop here
            break

        content_parts.append(boundary_chunk.content)
        total_tokens += chunk_tokens

    # Get sibling content if requested and we have budget
    sibling_content = []
    if include_siblings and total_tokens < max_tokens:
        sibling_ids = boundary_index.get_siblings(target_boundary_id)
        for sibling_id in sibling_ids:
            sibling_chunks = store.get_chunks_by_boundary(
                boundary_index,
                sibling_id,
                include_children=True
            )
            for sibling_chunk in sibling_chunks:
                chunk_tokens = sibling_chunk.metadata.token_count or count_tokens(sibling_chunk.content)
                if total_tokens + chunk_tokens > max_tokens:
                    break
                sibling_content.append(sibling_chunk.content)
                total_tokens += chunk_tokens

    # Get boundary metadata
    boundary = boundary_index.get_boundary(target_boundary_id)
    boundary_info = {
        "id": target_boundary_id,
        "type": boundary.type.value if boundary else None,
        "level": boundary.level if boundary else None,
        "title": boundary.title if boundary else None,
        "chunk_count": len(boundary_chunks),
    }

    # Combine content
    full_content = "\n\n".join(content_parts)
    if sibling_content:
        full_content += "\n\n---\n\n" + "\n\n".join(sibling_content)

    return {
        "content": full_content,
        "boundary_info": boundary_info,
        "token_count": total_tokens,
    }


# Tool schema for MCP registration
SEARCH_TOOL_SCHEMA = {
    "name": "knowledge_search",
    "description": "Search indexed documents using semantic and text similarity with optional boundary expansion",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to specific source names (optional)",
            },
            "top_k": {
                "type": "number",
                "default": 5,
                "description": "Number of results to return",
            },
            "include_context": {
                "type": "boolean",
                "default": True,
                "description": "Include surrounding chunks as context",
            },
            "expand_to_boundary": {
                "type": "string",
                "enum": ["chapter", "section", "subsection", "page"],
                "description": "Expand results to include full boundary content",
            },
            "max_return_tokens": {
                "type": "number",
                "default": 4096,
                "description": "Maximum tokens to return across all results",
            },
            "include_siblings": {
                "type": "boolean",
                "default": False,
                "description": "Include sibling boundaries when expanding",
            },
        },
        "required": ["query"],
    },
}
