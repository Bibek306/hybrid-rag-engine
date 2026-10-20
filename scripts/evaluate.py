import sys
import json
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
from app.config import settings
from app.search import search_engine


def is_ground_truth_match(chunk_data: dict, target_source: str, target_substring: str) -> bool:
    """Verifies if the retrieved chunk originates from the expected file and contains the expected fact."""
    source_match = chunk_data.get("source_file") == target_source
    content_match = target_substring.lower() in chunk_data.get("content", "").lower()
    return source_match and content_match


def run_dense_only(query: str, top_k: int) -> list[dict]:
    doc_ids = search_engine.retrieve_dense(query, top_k=top_k)
    return [search_engine.chunks[did] for did in doc_ids]


def run_sparse_only(query: str, top_k: int) -> list[dict]:
    doc_ids = search_engine.retrieve_sparse(query, top_k=top_k)
    return [search_engine.chunks[did] for did in doc_ids]


def run_hybrid_rerank(query: str, top_k: int) -> list[dict]:
    response = search_engine.search(
        query=query,
        first_stage_k=settings.DEFAULT_FIRST_STAGE_K,
        final_top_k=top_k
    )
    return [{"source_file": res.source_file, "content": res.content} for res in response.results]


def evaluate_pipeline(pipeline_fn, test_cases: list[dict], top_k: int) -> tuple[float, float, float]:
    """
    Computes:
      - Hit Rate@K: Proportion of queries with >= 1 correct chunk in top_k
      - MRR@K: Mean Reciprocal Rank (1/rank) of the first correct chunk
      - Latency: Average retrieval time per query in milliseconds
    """
    hits = 0
    reciprocal_ranks = []
    latencies = []

    for item in test_cases:
        query = item["query"]
        target_source = item["expected_source"]
        target_substring = item["target_substring"]

        start_time = time.perf_counter()
        retrieved_chunks = pipeline_fn(query, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(latency_ms)

        # Check rankings
        match_found = False
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_ground_truth_match(chunk, target_source, target_substring):
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                match_found = True
                break

        if not match_found:
            reciprocal_ranks.append(0.0)

    hit_rate = (hits / len(test_cases)) * 100.0
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))

    return hit_rate, mrr, avg_latency


def main():
    print("=" * 80)
    print("HYBRID RAG BENCHMARK: Hit Rate@K & Mean Reciprocal Rank (MRR)")
    print("=" * 80)

    # 1. Initialize engine
    search_engine.load_resources()

    # 2. Load golden dataset
    dataset_path = settings.EVAL_DATASET_PATH
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    k_val = settings.DEFAULT_FINAL_TOP_K
    print(f"Loaded {len(test_cases)} evaluation test cases from {dataset_path.name}")
    print(f"Evaluating top-K = {k_val}\n")

    pipelines = [
        ("Pure Dense (FAISS)", run_dense_only),
        ("Pure Sparse (BM25)", run_sparse_only),
        ("Hybrid (BM25 + FAISS + RRF + Rerank)", run_hybrid_rerank)
    ]

    print(f"{'Pipeline Strategy':<38} | {'Hit Rate@' + str(k_val):<12} | {'MRR@' + str(k_val):<10} | {'Avg Latency':<12}")
    print("-" * 80)

    for name, fn in pipelines:
        hit_rate, mrr, latency = evaluate_pipeline(fn, test_cases, top_k=k_val)
        print(f"{name:<38} | {hit_rate:>10.1f}% | {mrr:>10.3f} | {latency:>8.2f} ms")

    print("-" * 80)
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()