# Production-Grade Hybrid RAG Engine

A high-performance, two-stage Hybrid Retrieval-Augmented Generation (RAG) engine designed to solve the two fundamental failure modes of naive RAG: **vocabulary mismatch on exact tokens** and **semantic false positives from bi-encoder embeddings**.

Built using a modular machine learning lifecycle architecture featuring **FastAPI**, **Rank-BM25**, **FAISS**, **Sentence-Transformers**, and **Groq (Llama-3.3-70B)** for grounded synthesis with source citations.

---

## 🏛️ System Architecture: The Two-Stage Funnel

Production search operates on a two-stage retrieval funnel that balances high recall with microsecond precision: