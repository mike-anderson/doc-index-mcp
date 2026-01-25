"""
Input validation models for MCP Knowledge tools.

Provides Pydantic models with clear error messages that can be returned
via MCP to help agents adjust their inputs.
"""

import os
import re
from typing import Optional, Literal
from pydantic import BaseModel, field_validator, model_validator


# Valid source name pattern: alphanumeric, hyphens, underscores, dots
SOURCE_NAME_PATTERN = re.compile(r'^[\w\-\.]+$')

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf', '.docx', '.pptx', '.xlsx', '.xls'}
SUPPORTED_TABLE_FORMATS = {'.pdf', '.docx', '.xlsx', '.xls'}


class ValidationError(Exception):
    """Validation error with user-friendly message."""

    def __init__(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        self.message = message
        self.field = field
        self.suggestion = suggestion
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to dict for MCP error response."""
        result = {"error": self.message}
        if self.field:
            result["field"] = self.field
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


class IndexDocumentInput(BaseModel):
    """Validation for knowledge_index tool."""

    file_path: str
    source_name: Optional[str] = None

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "file_path is required and cannot be empty. "
                "Provide a path to the document, e.g., 'research/sources/document.pdf'"
            )

        v = v.strip()
        ext = os.path.splitext(v)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {supported}"
            )

        return v

    @field_validator('source_name')
    @classmethod
    def validate_source_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None

        v = v.strip()
        if not v:
            return None

        if not SOURCE_NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid source_name '{v}'. "
                "Source names can only contain letters, numbers, hyphens, underscores, and dots. "
                "Example: 'my-research-paper' or 'chapter_1.pdf'"
            )

        if len(v) > 100:
            raise ValueError(
                f"source_name is too long ({len(v)} chars). "
                "Maximum length is 100 characters."
            )

        return v


class SearchInput(BaseModel):
    """Validation for knowledge_search tool."""

    query: str
    sources: Optional[list[str]] = None
    top_k: int = 5
    include_context: bool = True
    expand_to_boundary: Optional[Literal['section', 'chapter', 'subsection', 'page']] = None
    max_return_tokens: int = 4096
    include_siblings: bool = False

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "query is required and cannot be empty. "
                "Provide a search query describing what you're looking for."
            )

        v = v.strip()
        if len(v) > 10000:
            raise ValueError(
                f"query is too long ({len(v)} chars). "
                "Maximum length is 10,000 characters. Try a more concise query."
            )

        return v

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"top_k must be at least 1, got {v}. "
                "Set top_k to the number of results you want (e.g., 5)."
            )
        if v > 100:
            raise ValueError(
                f"top_k cannot exceed 100, got {v}. "
                "For more results, make multiple searches with different queries."
            )
        return v

    @field_validator('max_return_tokens')
    @classmethod
    def validate_max_return_tokens(cls, v: int) -> int:
        if v < 100:
            raise ValueError(
                f"max_return_tokens must be at least 100, got {v}. "
                "This limits the total tokens returned across all results."
            )
        if v > 100000:
            raise ValueError(
                f"max_return_tokens cannot exceed 100,000, got {v}. "
                "Consider using a smaller limit and paginating results."
            )
        return v

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None

        validated = []
        for i, source in enumerate(v):
            if not source or not source.strip():
                raise ValueError(
                    f"sources[{i}] is empty. "
                    "Each source name must be non-empty. "
                    "Use knowledge_list to see available sources."
                )
            validated.append(source.strip())

        return validated if validated else None


class GetChunkInput(BaseModel):
    """Validation for knowledge_chunk tool."""

    chunk_id: str
    neighbors: int = 0

    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "chunk_id is required and cannot be empty. "
                "Chunk IDs have the format 'source_name:position' (e.g., 'my-doc:42')."
            )

        v = v.strip()
        if ':' not in v:
            raise ValueError(
                f"Invalid chunk_id format '{v}'. "
                "Chunk IDs must contain a colon, e.g., 'source_name:position'. "
                "Use knowledge_search to find valid chunk IDs."
            )

        return v

    @field_validator('neighbors')
    @classmethod
    def validate_neighbors(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"neighbors cannot be negative, got {v}. "
                "Set to 0 for just the chunk, or a positive number for surrounding chunks."
            )
        if v > 50:
            raise ValueError(
                f"neighbors cannot exceed 50, got {v}. "
                "For larger context, use expand_to_boundary in knowledge_search."
            )
        return v


class ReadDocumentInput(BaseModel):
    """Validation for read_document tool."""

    file_path: str
    max_chars: int = 100000
    include_metadata: bool = True

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "file_path is required and cannot be empty. "
                "Provide a path to the document, e.g., 'research/sources/report.pdf'"
            )

        v = v.strip()
        ext = os.path.splitext(v)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {supported}"
            )

        return v

    @field_validator('max_chars')
    @classmethod
    def validate_max_chars(cls, v: int) -> int:
        if v < 100:
            raise ValueError(
                f"max_chars must be at least 100, got {v}. "
                "This limits the maximum characters returned from the document."
            )
        if v > 10000000:
            raise ValueError(
                f"max_chars cannot exceed 10,000,000, got {v}. "
                "For very large documents, consider indexing with knowledge_index instead."
            )
        return v


class ListTablesInput(BaseModel):
    """Validation for list_tables tool."""

    file_path: str

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "file_path is required and cannot be empty. "
                "Provide a path to the document containing tables."
            )

        v = v.strip()
        ext = os.path.splitext(v)[1].lower()

        if ext not in SUPPORTED_TABLE_FORMATS:
            supported = ', '.join(sorted(SUPPORTED_TABLE_FORMATS))
            raise ValueError(
                f"Unsupported format '{ext}' for table extraction. "
                f"Supported formats: {supported}"
            )

        return v


class ExtractTableInput(BaseModel):
    """Validation for extract_table tool."""

    file_path: str
    table_index: int
    max_rows: Optional[int] = None
    include_headers: bool = True

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "file_path is required and cannot be empty. "
                "Provide a path to the document containing the table."
            )

        v = v.strip()
        ext = os.path.splitext(v)[1].lower()

        if ext not in SUPPORTED_TABLE_FORMATS:
            supported = ', '.join(sorted(SUPPORTED_TABLE_FORMATS))
            raise ValueError(
                f"Unsupported format '{ext}' for table extraction. "
                f"Supported formats: {supported}"
            )

        return v

    @field_validator('table_index')
    @classmethod
    def validate_table_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"table_index cannot be negative, got {v}. "
                "Table indices are 0-based. Use list_tables to see available tables."
            )
        return v

    @field_validator('max_rows')
    @classmethod
    def validate_max_rows(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1:
            raise ValueError(
                f"max_rows must be at least 1, got {v}. "
                "Omit this parameter to extract all rows."
            )
        if v > 100000:
            raise ValueError(
                f"max_rows cannot exceed 100,000, got {v}. "
                "For very large tables, consider extracting in batches."
            )
        return v


def validate_input(model_class: type[BaseModel], arguments: dict) -> dict:
    """
    Validate input arguments using a Pydantic model.

    Args:
        model_class: The Pydantic model class to use for validation
        arguments: The input arguments to validate

    Returns:
        Validated and cleaned arguments as a dict

    Raises:
        ValidationError: If validation fails, with helpful error message
    """
    try:
        validated = model_class(**arguments)
        return validated.model_dump(exclude_none=True)
    except Exception as e:
        # Extract the most useful error message
        if hasattr(e, 'errors'):
            # Pydantic validation error
            errors = e.errors()
            if errors:
                first_error = errors[0]
                field = '.'.join(str(loc) for loc in first_error.get('loc', []))
                msg = first_error.get('msg', str(e))
                raise ValidationError(msg, field=field)
        raise ValidationError(str(e))
