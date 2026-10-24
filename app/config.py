from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Hybrid RAG Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Paths
    DATA_DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    FAISS_INDEX_PATH: Path = STORAGE_DIR / "faiss_index.bin"
    BM25_STORE_PATH: Path = STORAGE_DIR / "bm25_store.pkl"
    EVAL_DATASET_PATH: Path = BASE_DIR / "data" / "golden_eval.json"

    # Neural Models
    DENSE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # Retrieval & Chunking Hyperparameters
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    DEFAULT_FIRST_STAGE_K: int = 15
    DEFAULT_FINAL_TOP_K: int = 3
    RRF_K: int = 60

    # LLM Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "openai/gpt-oss-120b"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()