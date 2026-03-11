# Knowledge Index MCP — Usage Guide

## Key Concept: Boundaries

Documents are chunked (~256 tokens) and embedded. The server also detects **boundaries** — structural markers (chapters, sections, subsections, pages) forming a hierarchy. Each chunk belongs to a boundary. **Siblings** are boundaries sharing the same parent.

## Search Patterns

**Basic search** — returns chunk fragments:
```
knowledge_search(query="authentication flow")
```

**Full section around a match** — use `expand_to_boundary`:
```
knowledge_search(query="error handling", expand_to_boundary="section", top_k=2, max_return_tokens=8192)
```

**Full chapter** — lower top_k, raise budget:
```
knowledge_search(query="data pipeline", expand_to_boundary="chapter", top_k=1, max_return_tokens=16384)
```

**Sibling sections** — see all peers under same parent:
```
knowledge_search(query="auth", expand_to_boundary="section", include_siblings=true, max_return_tokens=16384)
```

## When to Use What

| Need | Approach |
|---|---|
| Quick fact | Default search, top_k=3 |
| Full context around match | `expand_to_boundary="section"` |
| Whole topic area | `expand_to_boundary="chapter"`, top_k=1 |
| What else is in same chapter | `include_siblings=true` + expansion |
| Exact phrase | Just search — exact matches auto-boost |
| Specific table data | `list_tables` then `extract_table` |

## Critical: Token Budget

- Default 4096 = ~16 chunks, but only 1-2 sections
- With section expansion: use 8192+
- With chapter expansion: use 16384+
- With siblings: use 16384+
- Lower `top_k` when expanding (1-3)
