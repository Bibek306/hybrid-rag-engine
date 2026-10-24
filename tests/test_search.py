import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.search import search_engine


@pytest.fixture(scope="session", autouse=True)
def initialize_engine():
    """Ensure indices and neural models are loaded before running tests."""
    if not settings.FAISS_INDEX_PATH.exists() or not settings.BM25_STORE_PATH.exists():
        from scripts.indexer import build_indices
        build_indices()

    if not search_engine.is_ready:
        search_engine.load_resources()


@pytest.fixture(scope="module")
def client():
    """FastAPI test client fixture."""
    with TestClient(app) as test_client:
        yield test_client


# -------------------------------------------------------------
# Unit Tests: Core Retrieval & Math
# -------------------------------------------------------------
def test_reciprocal_rank_fusion_math():
    """
    Validates RRF ranking logic:
    A candidate present in both sparse and dense rankings must outrank
    a candidate appearing in only one list.
    """
    # Candidate 10 appears at rank 0 in both lists
    sparse_hits = [10, 20, 30]
    dense_hits = [10, 40, 50]

    fused = search_engine.reciprocal_rank_fusion(sparse_hits, dense_hits, rrf_k=60)

    # Candidate 10 receives 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0327
    assert fused[0] == 10, "Shared high-ranking candidate must place first in RRF"


def test_reciprocal_rank_fusion_tie_breaking():
    """Tests that candidate deduplication works across non-overlapping rankings."""
    sparse_hits = [1, 2]
    dense_hits = [3, 4]

    fused = search_engine.reciprocal_rank_fusion(sparse_hits, dense_hits, rrf_k=60)
    assert len(fused) == 4
    assert len(set(fused)) == 4


def test_search_engine_returns_valid_response():
    """Verifies internal search engine outputs compliant SearchResponse schema."""
    result = search_engine.search(
        query="CUDA_OUT_OF_MEMORY",
        first_stage_k=5,
        final_top_k=2
    )

    assert result.total_results <= 2
    assert result.latency_ms > 0
    assert len(result.results) > 0
    assert "troubleshooting.md" in result.results[0].source_file


# -------------------------------------------------------------
# Integration Tests: FastAPI Endpoints
# -------------------------------------------------------------
def test_health_check_endpoint(client):
    """Verifies /health reports healthy status and loaded chunks."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["index_loaded"] is True
    assert data["total_indexed_chunks"] > 0


def test_search_endpoint_success(client):
    """Tests valid POST /search payload."""
    payload = {
        "query": "How to resolve HTTP 429 rate limit?",
        "top_k": 3
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["results"]) <= 3
    assert "api_reference.md" in data["results"][0]["source_file"]
    assert "rerank_score" in data["results"][0]


def test_ask_endpoint_success(client):
    """Tests POST /ask end-to-end RAG synthesis endpoint."""
    payload = {
        "query": "How do I fix CUDA_OUT_OF_MEMORY errors?",
        "top_k": 2
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert data["total_sources"] <= 2
    assert len(data["sources"]) > 0


def test_search_endpoint_validation_errors(client):
    """Tests that invalid inputs trigger HTTP 422 before reaching models."""
    # Query too short (min length is 2)
    response_short = client.post("/search", json={"query": "a", "top_k": 3})
    assert response_short.status_code == 422

    # Top-K out of bounds (max is 20)
    response_out_of_bounds = client.post("/search", json={"query": "valid query", "top_k": 50})
    assert response_out_of_bounds.status_code == 422

    # Missing required query field
    response_missing_field = client.post("/search", json={"top_k": 3})
    assert response_missing_field.status_code == 422