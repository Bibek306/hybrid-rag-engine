from typing import Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
  """Represents a chunk extracted from a document suring indexing."""
  doc_id: int
  content: str
  source_file: str
  metadata: dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
  """Incoming search request payload."""
  query: str = Field(..., min_length=2, max_length=500, description="Search query string")
  top_k: int = Field(default=3, ge=1, le=20, description="Number of final results to return")

class SearchResultItem(BaseModel):
  """A single retrieved and reranked document chunk with citations."""
  doc_id: int
  source_file: str
  content: str
  rerank_score: float = Field(..., description="Cross-encoder relevance score")
  metadata: dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
  """Structured response returned by the /search endpoint."""
  query: str
  total_results: int
  latency_ms: float = Field(..., description="End-to-end search latency in milliseconds")
  results: list[SearchResultItem]

class HealthResponse(BaseModel):
  """Health status and index readiness response."""
  status: str
  index_loaded: bool
  total_indexed_chunks: int
  