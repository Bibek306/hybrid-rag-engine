from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
  APP_NAME: str = "Hybrid RAG ENGINE"
  APP_VERSION: str = "1.0.0"
  DEBUG: bool = False


  DATA_DOCS_DIR: Path = BASE_DIR / "data" / "docs"
  STORAGE_DIR: Path = BASE_DIR / "storage"
  FAISS_INDEX_PATH: Path = STORAGE_DIR / "faiss_index.bin"
  BM25_STORE_PATH: Path = STORAGE_DIR / "bm25_store.pkl"
  EVAL_DATASET_PATH: Path = BASE_DIR / "data" / "golden_eval.json"

  DENSE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
  RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

  CHUNK_SIZE: int = 300
  CHUNK_OVERLAP: int = 50

  DEFAULT_FIRST_STAGE_K: int = 15
  DEFAULT_FINAL_TOP_K: int = 3
  RRF_K: int = 60

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"
  )

settings = Settings()  