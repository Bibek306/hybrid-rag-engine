import os
import time
from pathlib import Path
from dotenv import load_dotenv

# 1. Load local .env (if present)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

import streamlit as st

# 2. Bridge Streamlit Community Cloud Secrets to environment variables
if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.config import settings
from app.generator import generator
from app.search import search_engine

# Page setup
st.set_page_config(
    page_title="Hybrid RAG Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Production-Grade Hybrid RAG Engine")
st.caption(
    f"Two-Stage Retrieval: BM25 + FAISS → Reciprocal Rank Fusion (k={settings.RRF_K}) "
    f"→ BGE Cross-Encoder → Groq ({generator.model_name})"
)

# Initialize models and indices once with fallback generation
@st.cache_resource(show_spinner="Loading indices and neural models into memory...")
def init_engine():
    # If storage files do not exist yet, build them on the fly
    if not settings.FAISS_INDEX_PATH.exists() or not settings.BM25_STORE_PATH.exists():
        from scripts.indexer import build_indices
        build_indices()

    search_engine.load_resources()
    return search_engine

engine = init_engine()

# Sidebar configuration
with st.sidebar:
    st.header("Search Configuration")
    top_k = st.slider("Top Sources (k)", min_value=1, max_value=10, value=settings.DEFAULT_FINAL_TOP_K)
    st.markdown("---")
    st.markdown("**Architecture Specs:**")
    st.markdown("- **Sparse:** BM25 (Rank-BM25)")
    st.markdown(f"- **Dense:** FAISS ({settings.DENSE_MODEL_NAME})")
    st.markdown(f"- **Fusion:** Reciprocal Rank Fusion ($k={settings.RRF_K}$)")
    st.markdown(f"- **Reranker:** Cross-Encoder ({settings.RERANKER_MODEL_NAME})")
    st.markdown(f"- **Generation:** Groq ({generator.model_name})")

# Search and Ask input
query = st.text_input("Ask a technical question about your documents:", placeholder="e.g., How do I fix CUDA_OUT_OF_MEMORY errors?")

if st.button("Search & Generate", type="primary") and query.strip():
    start_time = time.perf_counter()
    
    with st.spinner("Retrieving relevant passages and synthesizing answer..."):
        # 1. Two-stage hybrid search
        search_res = engine.search(query=query, final_top_k=top_k)
        
        # 2. LLM synthesis
        answer = generator.generate_answer(query=query, context_chunks=search_res.results)
        
        total_time = (time.perf_counter() - start_time) * 1000

    # Display Answer
    st.subheader("💡 Answer")
    st.markdown(answer)
    st.caption(f"End-to-end latency: **{total_time:.2f} ms** | Retrieved sources: **{len(search_res.results)}**")

    st.markdown("---")
    
    # Display Retrieved Sources & Rerank Scores
    st.subheader("📚 Verified Retrieved Context")
    for i, chunk in enumerate(search_res.results, 1):
        with st.expander(f"Source {i}: `{chunk.source_file}` (Cross-Encoder Score: {chunk.rerank_score:.4f})"):
            st.markdown(chunk.content)