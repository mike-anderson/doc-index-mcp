"""
MCP Knowledge Server

A Model Context Protocol server for knowledge management with
boundary-aware document chunking and semantic search.
"""

__all__ = ["KnowledgeServer", "main"]
__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import for server components that require mcp dependency."""
    if name == "KnowledgeServer":
        from .server import KnowledgeServer
        return KnowledgeServer
    elif name == "main":
        from .server import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
