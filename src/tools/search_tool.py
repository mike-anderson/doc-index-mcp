"""
Search tool for the mcp-knowledge MCP server.

Provides semantic search across indexed documents with boundary-aware
expansion capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional

from services.chunker import Chunk, BoundaryIndex, count_tokens
from services.vector_store import VectorStore, SearchResult
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
    Execute a semantic search with optional boundary expansion.

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

    # Collect results from all relevant stores
    all_results: list[tuple[SearchResult, str]] = []  # (result, source_name)

    for source_name, store in stores.items():
        # Filter by sources if specified
        if params.sources and source_name not in params.sources:
            continue

        results = store.search(
            query_embedding=query_embedding,
            k=params.top_k * 2,  # Fetch extra for filtering/merging
        )

        for result in results:
            all_results.append((result, source_name))

    # Sort by score and take top_k
    all_results.sort(key=lambda x: -x[0].score)
    top_results = all_results[:params.top_k]

    # Build response items
    response = SearchResponse(query=params.query)
    total_tokens = 0

    for result, source_name in top_results:
        chunk = result.chunk
        item = SearchResultItem(
            chunk_id=chunk.id,
            source_name=source_name,
            score=result.score,
            content=chunk.content,
            metadata={
                "position": chunk.metadata.position,
                "total_chunks": chunk.metadata.total_chunks,
                "boundary_type": chunk.metadata.boundary_type,
                "boundary_id": chunk.metadata.boundary_id,
                "boundary_title": chunk.metadata.boundary_title,
                "token_count": chunk.metadata.token_count,
            },
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
    "description": "Search indexed documents using semantic similarity with optional boundary expansion",
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
