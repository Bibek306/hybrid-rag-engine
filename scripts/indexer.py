import pickle
import sys
from pathlib import Path

# Add project root to sys.path so local app packages import first
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import DocumentChunk


def build_indices():
    docs_dir = settings.DATA_DOCS_DIR
    storage_dir = settings.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    if not docs_dir.exists():
        print(f"Error: Documents directory not found at {docs_dir}")
        sys.exit(1)

    # 1. Discover all source files
    doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"Error: No documents found in {docs_dir}. Place markdown or text files there first.")
        sys.exit(1)

    print(f"Discovered {len(doc_files)} document(s) in {docs_dir}:")
    for f in doc_files:
        print(f" - {f.name}")

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks: list[DocumentChunk] = []
    chunk_id = 0

    for file_path in doc_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        splits = splitter.split_text(text)
        for split in splits:
            cleaned_text = split.strip()
            if cleaned_text:
                chunks.append(
                    DocumentChunk(
                        doc_id=chunk_id,
                        content=cleaned_text,
                        source_file=file_path.name,
                        metadata={"length": len(cleaned_text)}
                    )
                )
                chunk_id += 1

    if not chunks:
        print("Error: No text content found across document files to chunk.")
        sys.exit(1)

    print(f"\nGenerated {len(chunks)} total text chunks.")

    # 3. Dense Indexing (FAISS)
    print(f"Loading dense embedding model: {settings.DENSE_MODEL_NAME}...")
    encoder = SentenceTransformer(settings.DENSE_MODEL_NAME)
    corpus_texts = [c.content for c in chunks]

    print("Computing embeddings (with L2 normalization for Inner Product)...")
    embeddings = encoder.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    # IndexFlatIP calculates inner product, which equals cosine similarity on normalized vectors
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    faiss_path = str(settings.FAISS_INDEX_PATH)
    faiss.write_index(faiss_index, faiss_path)
    print(f"Saved FAISS index ({faiss_index.ntotal} vectors) to: {faiss_path}")

    # 4. Sparse Indexing Data (BM25 Payload)
    print("Preparing BM25 sparse tokenized corpus...")
    tokenized_corpus = [c.content.lower().split() for c in chunks]

    # Save chunks and tokenized corpus together so retrieval mapping is 1:1 by doc_id
    bm25_payload = {
        "chunks": [c.model_dump() for c in chunks],
        "tokenized": tokenized_corpus
    }

    bm25_path = settings.BM25_STORE_PATH
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_payload, f)
    print(f"Saved BM25 payload to: {bm25_path}")

    print("\nOffline indexing completed successfully.")


if __name__ == "__main__":
    build_indices()