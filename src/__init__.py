"""
MCP Doc Index Server

A Model Context Protocol server for document indexing with
boundary-aware chunking and semantic search.
"""

__all__ = ["DocIndexServer", "main"]
__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import for server components that require mcp dependency."""
    if name == "DocIndexServer":
        from .server import DocIndexServer
        return DocIndexServer
    elif name == "main":
        from .server import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
