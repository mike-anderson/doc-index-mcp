"""
MCP Knowledge Index Server

Provides semantic search over indexed documents with boundary-aware chunking.

Tools:
- knowledge_index: Index a document (PDF, TXT, MD, DOCX, PPTX, XLSX)
- knowledge_search: Semantic and text search with boundary expansion
- knowledge_list: List indexed sources
- knowledge_chunk: Retrieve specific chunks
- read_document: Read documents without indexing (PDF, Word, PowerPoint, Excel)
- list_tables: List all tables in a document
- extract_table: Extract a specific table as CSV
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Direct imports - no lazy loading
from services import document_loader
from services import chunker
from services import vector_store
from services import table_extractor
from services.embedder import Embedder
from tools import search_tool
from validation import (
    validate_input,
    ValidationError,
    IndexDocumentInput,
    SearchInput,
    GetChunkInput,
    ReadDocumentInput,
    ListTablesInput,
    ExtractTableInput,
)

# Supported file extensions
SUPPORTED_EXTENSIONS = [".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx", ".xlsx", ".xls"]
SUPPORTED_TABLE_FORMATS = [".pdf", ".docx", ".xlsx", ".xls"]

# Search tool schema
SEARCH_TOOL_SCHEMA = {
    "name": "knowledge_search",
    "description": "Search indexed documents using semantic and text similarity with optional boundary expansion",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of source names to search (searches all if not specified)",
            },
            "top_k": {
                "type": "number",
                "default": 5,
                "description": "Number of results to return",
            },
            "include_context": {
                "type": "boolean",
                "default": True,
                "description": "Include surrounding context snippets",
            },
            "expand_to_boundary": {
                "type": "string",
                "enum": ["section", "chapter"],
                "description": "Expand results to include full section or chapter",
            },
            "max_return_tokens": {
                "type": "number",
                "default": 4096,
                "description": "Maximum tokens (tiktoken cl100k_base) to return across all results. Chunks are measured in tokens.",
            },
            "include_siblings": {
                "type": "boolean",
                "default": False,
                "description": "Include sibling sections when expanding to boundary",
            },
        },
        "required": ["query"],
    },
}


class KnowledgeServer:
    """MCP server for knowledge management."""

    def __init__(self, knowledge_dir: str = ".knowledge"):
        """
        Initialize the knowledge server.

        Args:
            knowledge_dir: Directory for storing indices and metadata.
                          Default is .knowledge in current directory
        """
        # Working directory for resolving relative file paths
        # Set via MCP_WORKING_DIR env var (should match Claude's cwd)
        self.working_dir = os.environ.get("MCP_WORKING_DIR", os.getcwd())

        # Knowledge dir defaults to .knowledge in working directory
        self.knowledge_dir = os.environ.get("KNOWLEDGE_DIR", knowledge_dir)
        if not os.path.isabs(self.knowledge_dir):
            self.knowledge_dir = os.path.join(self.working_dir, self.knowledge_dir)

        self.server = Server("knowledge-index-mcp")

        # Storage
        self.stores: dict[str, vector_store.VectorStore] = {}
        self.boundary_indices: dict[str, chunker.BoundaryIndex] = {}
        self.manifest: dict = {"sources": {}}

        # Initialize embedder
        self._embedder = Embedder()

        # Setup handlers
        self._setup_handlers()

    def _resolve_path(self, file_path: str) -> str:
        """Resolve a file path relative to the working directory.

        Args:
            file_path: Path that may be relative or absolute

        Returns:
            Absolute path resolved relative to working_dir if needed

        Raises:
            ValueError: If path attempts to escape the working directory
        """
        # Normalize the working directory
        working_dir_normalized = os.path.normpath(self.working_dir)

        if os.path.isabs(file_path):
            # Absolute paths must be within working directory
            resolved = os.path.normpath(file_path)
            if not resolved.startswith(working_dir_normalized + os.sep) and resolved != working_dir_normalized:
                raise ValueError(f"Path must be within {self.working_dir}")
            return resolved

        # Resolve relative path and check for traversal
        resolved = os.path.normpath(os.path.join(working_dir_normalized, file_path))
        if not resolved.startswith(working_dir_normalized + os.sep) and resolved != working_dir_normalized:
            raise ValueError(f"Path traversal not allowed: {file_path}")
        return resolved

    def _setup_handlers(self):
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="knowledge_index",
                    description="Index a document for semantic search with boundary-aware chunking",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": f"Path to the document file. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
                            },
                            "source_name": {
                                "type": "string",
                                "description": "Optional name for this source (defaults to filename)",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(**SEARCH_TOOL_SCHEMA),
                Tool(
                    name="knowledge_list",
                    description="List all indexed knowledge sources",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="knowledge_chunk",
                    description="Retrieve specific chunk(s) by ID with optional neighbors",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "ID of the chunk to retrieve",
                            },
                            "neighbors": {
                                "type": "number",
                                "default": 0,
                                "description": "Number of neighboring chunks to include on each side",
                            },
                        },
                        "required": ["chunk_id"],
                    },
                ),
                Tool(
                    name="read_document",
                    description="Read and extract text from documents (PDF, Word, PowerPoint, Excel). Returns formatted text optimized for LLM consumption without indexing.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": f"Path to the document file. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
                            },
                            "max_chars": {
                                "type": "number",
                                "default": 100000,
                                "description": "Maximum characters (not tokens) to return. Large documents are truncated at this limit.",
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include document metadata in response",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in a document. Returns table index, location, headers, and row count for each table found.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": f"Path to the document file. Supported formats: {', '.join(SUPPORTED_TABLE_FORMATS)}",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="extract_table",
                    description="Extract a specific table from a document as CSV. Use list_tables first to discover available tables.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": f"Path to the document file. Supported formats: {', '.join(SUPPORTED_TABLE_FORMATS)}",
                            },
                            "table_index": {
                                "type": "number",
                                "description": "Index of the table to extract (0-based, from list_tables)",
                            },
                            "max_rows": {
                                "type": "number",
                                "description": "Maximum number of data rows to extract (optional, extracts all if not specified)",
                            },
                            "include_headers": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include header row in CSV output",
                            },
                        },
                        "required": ["file_path", "table_index"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            try:
                if name == "knowledge_index":
                    validated = validate_input(IndexDocumentInput, arguments)
                    result = await self._index_document(
                        file_path=validated["file_path"],
                        source_name=validated.get("source_name"),
                    )
                elif name == "knowledge_search":
                    validated = validate_input(SearchInput, arguments)
                    result = await self._search(validated)
                elif name == "knowledge_list":
                    result = await self._list_sources()
                elif name == "knowledge_chunk":
                    validated = validate_input(GetChunkInput, arguments)
                    result = await self._get_chunk(
                        chunk_id=validated["chunk_id"],
                        neighbors=validated.get("neighbors", 0),
                    )
                elif name == "read_document":
                    validated = validate_input(ReadDocumentInput, arguments)
                    result = await self._read_document(
                        file_path=validated["file_path"],
                        max_chars=validated.get("max_chars", 100000),
                        include_metadata=validated.get("include_metadata", True),
                    )
                elif name == "list_tables":
                    validated = validate_input(ListTablesInput, arguments)
                    result = await self._list_tables(
                        file_path=validated["file_path"],
                    )
                elif name == "extract_table":
                    validated = validate_input(ExtractTableInput, arguments)
                    result = await self._extract_table(
                        file_path=validated["file_path"],
                        table_index=validated["table_index"],
                        max_rows=validated.get("max_rows"),
                        include_headers=validated.get("include_headers", True),
                    )
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except ValidationError as e:
                return [TextContent(type="text", text=json.dumps(e.to_dict(), indent=2))]
            except FileNotFoundError as e:
                return [TextContent(type="text", text=json.dumps({
                    "error": f"File not found: {e.filename if hasattr(e, 'filename') else str(e)}",
                    "suggestion": "Check that the file path is correct and the file exists."
                }, indent=2))]
            except PermissionError as e:
                return [TextContent(type="text", text=json.dumps({
                    "error": f"Permission denied: {str(e)}",
                    "suggestion": "Check file permissions or try a different file."
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async def _index_document(
        self,
        file_path: str,
        source_name: Optional[str] = None,
    ) -> dict:
        """Index a document for search with boundary-aware chunking."""
        # Resolve path relative to working directory
        resolved_path = self._resolve_path(file_path)

        # Load document
        doc = await document_loader.load_document(resolved_path)
        source = document_loader.get_source_name(resolved_path, source_name)

        # Chunk the document with boundary-aware chunking
        options = chunker.ChunkOptions()
        chunks, boundary_index = chunker.chunk_document(doc.content, source, options)
        self.boundary_indices[source] = boundary_index

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedder.embed_texts(texts)

        # Create and populate vector store
        store = vector_store.VectorStore(dimension=self._embedder.dimension)
        await store.initialize(max_elements=max(10000, len(chunks) * 2))
        await store.add_chunks(chunks, embeddings)
        self.stores[source] = store

        # Save to disk
        await self._save_source(source, boundary_index)

        # Update manifest
        self.manifest["sources"][source] = {
            "path": file_path,
            "chunks": len(chunks),
            "indexed_at": datetime.now().isoformat(),
            "file_type": doc.file_type,
            "has_boundaries": len(boundary_index.boundaries) > 0,
        }
        await self._save_manifest()

        return {
            "success": True,
            "source_name": source,
            "chunks_created": len(chunks),
            "boundaries_detected": len(boundary_index.boundaries),
        }

    async def _search(self, arguments: dict) -> dict:
        """Execute a search query."""
        # Ensure stores are loaded from disk (in case server restarted)
        await self._ensure_stores_loaded()

        params = search_tool.SearchParams(
            query=arguments["query"],
            sources=arguments.get("sources"),
            top_k=arguments.get("top_k", 5),
            include_context=arguments.get("include_context", True),
            expand_to_boundary=arguments.get("expand_to_boundary"),
            max_return_tokens=arguments.get("max_return_tokens", 4096),
            include_siblings=arguments.get("include_siblings", False),
        )

        response = await search_tool.execute_search(
            params=params,
            stores=self.stores,
            boundary_indices=self.boundary_indices,
            embed_fn=self._embedder.embed_text,
        )

        return {
            "query": response.query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "source": r.source_name,
                    "score": round(r.score, 4),
                    "content": r.content,
                    "metadata": r.metadata,
                    "context": r.context,
                    "expanded_content": r.expanded_content,
                    "boundary_info": r.boundary_info,
                    "text_score": round(r.text_score, 4) if r.text_score else None,
                    "match_type": r.match_type,
                }
                for r in response.results
            ],
            "total_tokens": response.total_tokens,
            "expansion_applied": response.expansion_applied,
        }

    async def _list_sources(self) -> dict:
        """List all indexed sources."""
        return {
            "sources": self.manifest.get("sources", {}),
            "total_sources": len(self.manifest.get("sources", {})),
        }

    async def _get_chunk(self, chunk_id: str, neighbors: int = 0) -> dict:
        """Get a specific chunk with optional neighbors."""
        # Find the source from chunk_id
        source = chunk_id.split(":")[0] if ":" in chunk_id else None

        if source and source in self.stores:
            store = self.stores[source]

            if neighbors > 0:
                chunks = store.get_chunk_with_neighbors(chunk_id, n=neighbors)
                return {
                    "chunks": [
                        {
                            "id": c.id,
                            "content": c.content,
                            "metadata": c.metadata.to_dict(),
                        }
                        for c in chunks
                    ]
                }
            else:
                chunk = store.get_chunk_by_id(chunk_id)
                if chunk:
                    return {
                        "chunk": {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": chunk.metadata.to_dict(),
                        }
                    }

        return {"error": f"Chunk not found: {chunk_id}"}

    async def _read_document(
        self,
        file_path: str,
        max_chars: int = 100000,
        include_metadata: bool = True,
    ) -> dict:
        """Read a document and return its content without indexing."""
        # Resolve path relative to working directory
        resolved_path = self._resolve_path(file_path)

        doc = await document_loader.load_document(resolved_path)

        content = doc.content
        truncated = False

        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        result = {
            "content": content,
            "file_type": doc.file_type,
            "truncated": truncated,
            "char_count": len(content),
        }

        if include_metadata:
            result["metadata"] = {
                **doc.metadata,
                "structure": doc.structure,
            }

        return result

    async def _list_tables(self, file_path: str) -> dict:
        """List all tables in a document."""
        # Resolve path relative to working directory
        resolved_path = self._resolve_path(file_path)

        tables = await table_extractor.list_tables(resolved_path)

        return {
            "file_path": file_path,
            "table_count": len(tables),
            "tables": [
                {
                    "index": t.index,
                    "location": t.location,
                    "headers": t.headers,
                    "row_count": t.row_count,
                    "col_count": t.col_count,
                }
                for t in tables
            ],
        }

    async def _extract_table(
        self,
        file_path: str,
        table_index: int,
        max_rows: Optional[int] = None,
        include_headers: bool = True,
    ) -> dict:
        """Extract a specific table as CSV."""
        # Resolve path relative to working directory
        resolved_path = self._resolve_path(file_path)

        table = await table_extractor.extract_table(resolved_path, table_index, max_rows)
        csv_content = table_extractor.table_to_csv(table, include_headers=include_headers)

        return {
            "file_path": file_path,
            "table_index": table_index,
            "location": table.location,
            "headers": table.headers,
            "row_count": table.row_count,
            "col_count": table.col_count,
            "csv": csv_content,
        }

    async def _save_source(self, source_name: str, boundary_index: Any):
        """Save a source's vector store and boundary index to disk."""
        source_dir = os.path.join(self.knowledge_dir, "vectors", source_name)
        os.makedirs(source_dir, exist_ok=True)

        # Save vector store
        store = self.stores.get(source_name)
        if store:
            await store.save(source_dir)

        # Save boundary index
        if boundary_index and boundary_index.boundaries:
            boundary_path = os.path.join(source_dir, "boundaries.json")
            boundary_index.save(boundary_path)

    async def _save_manifest(self):
        """Save manifest to disk."""
        os.makedirs(self.knowledge_dir, exist_ok=True)
        manifest_path = os.path.join(self.knowledge_dir, "manifest.json")

        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    async def load_existing_stores(self):
        """Load existing indices from disk on startup."""
        manifest_path = os.path.join(self.knowledge_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            return

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

    async def _ensure_stores_loaded(self):
        """Load vector stores from disk (called when search is needed)."""
        # Load each source that isn't already loaded
        for source_name, source_info in self.manifest.get("sources", {}).items():
            if source_name in self.stores:
                continue

            source_dir = os.path.join(self.knowledge_dir, "vectors", source_name)

            if not os.path.exists(source_dir):
                continue

            # Load vector store
            store = vector_store.VectorStore(dimension=self._embedder.dimension)
            try:
                await store.load(source_dir)
                self.stores[source_name] = store
            except Exception as e:
                print(f"Warning: Failed to load store for {source_name}: {e}")
                continue

            # Load boundary index
            boundary_path = os.path.join(source_dir, "boundaries.json")
            if os.path.exists(boundary_path):
                try:
                    self.boundary_indices[source_name] = chunker.BoundaryIndex.load(boundary_path)
                except Exception as e:
                    print(f"Warning: Failed to load boundaries for {source_name}: {e}")
                    self.boundary_indices[source_name] = chunker.BoundaryIndex()
            else:
                self.boundary_indices[source_name] = chunker.BoundaryIndex()

    async def run(self):
        """Run the MCP server."""
        await self.load_existing_stores()

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def main():
    """Entry point for the MCP server."""
    server = KnowledgeServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
