# Production-Grade Hybrid Search & Retrieval Engine (RAG)

A two-stage, low-latency document retrieval engine engineered in Python. It pairs **Lexical Search (BM25)** and **Dense Vector Search (FAISS)** using **Reciprocal Rank Fusion (RRF)** for high-recall candidate generation, followed by a **Cross-Encoder Neural Reranker** for high-precision answer filtering.

Built with an offline ETL indexer, singleton model caching via FastAPI `lifespan`, strict Pydantic v2 schemas, automated evaluation benchmarks, and containerized deployment.

---

## 📌 Architecture Overview

Naive RAG pipelines rely strictly on bi-encoder cosine similarity, routinely failing on exact tokens (error codes, HTTP statuses, function names) and suffering from semantic false positives. This system solves both problems using a two-stage retrieval design: