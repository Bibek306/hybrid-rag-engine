import logging
import pickle
import time
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings
from app.schemas import SearchResponse, SearchResultItem

logger = logging.getLogger("hybrid-rag-engine")


class HybridSearchEngine:
    """
    Two-stage retrieval engine:
      Stage 1: Dual candidate retrieval via BM25 (lexical) and FAISS (semantic)
      Stage 2: Candidate combination via Reciprocal Rank Fusion (RRF)
      Stage 3: High-precision reranking via Cross-Encoder
    """

    def __init__(self):
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25_index: BM25Okapi | None = None
        self.chunks: list[dict] = []
        self.bi_encoder: SentenceTransformer | None = None
        self.reranker: CrossEncoder | None = None
        self._is_ready: bool = False

    def load_resources(self):
        """Loads pre-built indices and neural models into memory."""
        logger.info("Initializing search engine resources...")

        # 1. Load BM25 tokenized corpus and raw chunks
        if not settings.BM25_STORE_PATH.exists():
            raise FileNotFoundError(
                f"Missing BM25 store at {settings.BM25_STORE_PATH}. Run scripts/indexer.py first."
            )
        with open(settings.BM25_STORE_PATH, "rb") as f:
            bm25_payload = pickle.load(f)

        self.chunks = bm25_payload["chunks"]
        self.bm25_index = BM25Okapi(bm25_payload["tokenized"])

        # 2. Load FAISS dense index
        if not settings.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Missing FAISS index at {settings.FAISS_INDEX_PATH}. Run scripts/indexer.py first."
            )
        self.faiss_index = faiss.read_index(str(settings.FAISS_INDEX_PATH))

        # 3. Load neural models
        logger.info(f"Loading Bi-Encoder: {settings.DENSE_MODEL_NAME}...")
        self.bi_encoder = SentenceTransformer(settings.DENSE_MODEL_NAME)

        logger.info(f"Loading Cross-Encoder: {settings.RERANKER_MODEL_NAME}...")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)

        self._is_ready = True
        logger.info(f"Engine ready with {len(self.chunks)} chunks loaded.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def retrieve_sparse(self, query: str, top_k: int) -> list[int]:
        """BM25 keyword search: returns list of doc_ids ordered by score."""
        tokens = query.lower().split()
        if not tokens or not self.bm25_index:
            return []

        scores = self.bm25_index.get_scores(tokens)
        safe_k = min(top_k, len(self.chunks))
        return np.argsort(scores)[::-1][:safe_k].tolist()

    def retrieve_dense(self, query: str, top_k: int) -> list[int]:
        """FAISS dense vector search: returns list of doc_ids ordered by cosine similarity."""
        if not self.bi_encoder or not self.faiss_index:
            return []

        query_vec = self.bi_encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        safe_k = min(top_k, len(self.chunks))
        _, indices = self.faiss_index.search(query_vec, safe_k)
        
        # Filter out invalid FAISS indices (-1 is returned if fewer results exist)
        return [idx for idx in indices[0].tolist() if 0 <= idx < len(self.chunks)]

    def reciprocal_rank_fusion(
        self,
        sparse_hits: list[int],
        dense_hits: list[int],
        rrf_k: int = settings.RRF_K
    ) -> list[int]:
        """
        Combines two ranked candidate lists using RRF:
        RRF_Score(d) = sum(1 / (k + rank + 1))
        """
        rrf_scores: dict[int, float] = {}

        # Accumulate sparse ranks (1-based rank indexing)
        for rank, doc_id in enumerate(sparse_hits):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Accumulate dense ranks
        for rank, doc_id in enumerate(dense_hits):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort candidate IDs by descending RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_candidates]

    def search(
        self,
        query: str,
        first_stage_k: int = settings.DEFAULT_FIRST_STAGE_K,
        final_top_k: int = settings.DEFAULT_FINAL_TOP_K
    ) -> SearchResponse:
        """Executes full hybrid search pipeline: Sparse + Dense -> RRF -> Cross-Encoder Rerank."""
        if not self._is_ready:
            raise RuntimeError("HybridSearchEngine is not initialized. Call load_resources() first.")

        start_time = time.perf_counter()

        # Step 1: Stage 1 candidate retrieval
        sparse_candidates = self.retrieve_sparse(query, top_k=first_stage_k)
        dense_candidates = self.retrieve_dense(query, top_k=first_stage_k)

        # Step 2: Reciprocal Rank Fusion
        fused_candidate_ids = self.reciprocal_rank_fusion(sparse_candidates, dense_candidates)
        candidate_subset = fused_candidate_ids[:first_stage_k]

        # Early return if no candidates match
        if not candidate_subset:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=query,
                total_results=0,
                latency_ms=round(latency_ms, 2),
                results=[]
            )

        # Step 3: Stage 2 Cross-Encoder Reranking
        pairs = [[query, self.chunks[doc_id]["content"]] for doc_id in candidate_subset]
        cross_scores = self.reranker.predict(pairs)

        # Step 4: Final top_k sorting
        ranked_indices = np.argsort(cross_scores)[::-1][:final_top_k]

        results: list[SearchResultItem] = []
        for idx in ranked_indices:
            doc_id = candidate_subset[idx]
            chunk_data = self.chunks[doc_id]
            results.append(
                SearchResultItem(
                    doc_id=doc_id,
                    source_file=chunk_data["source_file"],
                    content=chunk_data["content"],
                    rerank_score=float(cross_scores[idx]),
                    metadata=chunk_data.get("metadata", {})
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResponse(
            query=query,
            total_results=len(results),
            latency_ms=round(latency_ms, 2),
            results=results
        )


# Singleton instance ready to be imported across the application
search_engine = HybridSearchEngine()