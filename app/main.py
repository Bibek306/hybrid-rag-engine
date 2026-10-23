import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import SearchRequest, SearchResponse, HealthResponse
from app.search import search_engine

import time
from app.generator import generator
from app.schemas import AskResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid-rag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context: Runs initialization on startup
    and clean-up on shutdown.
    """
    logger.info("Initializing Hybrid Search Engine and loading models...")
    try:
        search_engine.load_resources()
        logger.info(
            f"Ready: Indexed chunks={search_engine.total_chunks}, "
            f"Dense Model={settings.DENSE_MODEL_NAME}, "
            f"Reranker={settings.RERANKER_MODEL_NAME}"
        )
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: {e}")
        logger.error("Ensure you ran 'python scripts/indexer.py' before launching the API.")
        raise e

    yield

    logger.info("Shutting down Hybrid Search API service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Hybrid Search (BM25 + FAISS + RRF + Cross-Encoder Reranker)",
    lifespan=lifespan
)

# Enable CORS for local dashboards or frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and Index Readiness Check"
)
async def health_check():
    """Returns system status and index readiness."""
    return HealthResponse(
        status="healthy" if search_engine.is_ready else "uninitialized",
        index_loaded=search_engine.is_ready,
        total_indexed_chunks=search_engine.total_chunks
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Search"],
    summary="Execute Hybrid Search with Cross-Encoder Reranking"
)
async def execute_search(payload: SearchRequest):
    """
    Executes a two-stage hybrid search:
      1. BM25 (lexical) + FAISS (semantic) retrieval
      2. Reciprocal Rank Fusion (RRF)
      3. Cross-Encoder reranking
    """
    if not search_engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search engine is not initialized or indices are missing."
        )

    try:
        response = search_engine.search(
            query=payload.query,
            first_stage_k=settings.DEFAULT_FIRST_STAGE_K,
            final_top_k=payload.top_k
        )
        logger.info(
            f"Query: '{payload.query}' | Results: {response.total_results} | Latency: {response.latency_ms}ms"
        )
        return response
    except Exception as e:
        logger.error(f"Search execution error for query '{payload.query}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the search pipeline."
        )

@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Pipeline"],
    summary="End-to-End RAG: Hybrid Search + Neural Reranking + LLM Synthesis"
)
async def ask_question(payload: SearchRequest):
    """
    Executes full End-to-End RAG:
      1. Stage 1: BM25 + FAISS retrieval
      2. Fusion: Reciprocal Rank Fusion (RRF)
      3. Stage 2: Cross-Encoder reranking
      4. Generation: LLM synthesizes cited response
    """
    if not search_engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search engine is not initialized."
        )

    start_time = time.perf_counter()

    # 1. Retrieve & rerank relevant chunks
    search_res = search_engine.search(
        query=payload.query,
        final_top_k=payload.top_k
    )

    # 2. Synthesize answer with LLM
    answer_text = generator.generate_answer(
        query=payload.query,
        context_chunks=search_res.results
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return AskResponse(
        query=payload.query,
        answer=answer_text,
        total_sources=len(search_res.results),
        latency_ms=round(latency_ms, 2),
        sources=search_res.results
    )    