# Production-Grade Hybrid RAG Engine

A high-performance, two-stage Hybrid Retrieval-Augmented Generation (RAG) engine designed to systematically eliminate the twin failure modes of naive vector search: **vocabulary mismatch on exact tokens** and **semantic false positives from approximate nearest-neighbor embeddings**.

Built using a modular machine learning architecture featuring **FastAPI**, **Rank-BM25**, **FAISS**, **Sentence-Transformers (BGE)**, and **Groq** for low-latency, cited answer synthesis.

---

## 🏛️ System Architecture: The Two-Stage Funnel

Production information retrieval balances broad recall across heterogeneous query types with microsecond ranking precision: