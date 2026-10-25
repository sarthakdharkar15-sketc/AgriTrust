"""Build persistent Chroma vector collection and BM25 index for AgriTrust hybrid retrieval."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agritrust.build_index")


def build_bm25_index(chunks: List[Dict[str, Any]], output_path: str = "data/index/bm25.pkl"):
    """Tokenize and build BM25Okapi index, persisting it to disk."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = []
    for chk in chunks:
        # Tokenize by simple lowercased word splitting
        tokens = [w.strip(".,;:?!()[]'\"").lower() for w in chk["text"].split() if w.strip()]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Persisted BM25 index with {len(tokenized_corpus)} documents to {out_file}")
    return bm25


def build_chroma_collection(
    chunks: List[Dict[str, Any]],
    db_dir: str = "data/index/chroma",
    collection_name: str = "agritrust_corpus",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """Build or populate Chroma persistent vector collection with dense embeddings."""
    import chromadb
    from chromadb.config import Settings

    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    # Reset collection if exists to ensure freshness
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Compute embeddings
    texts = [chk["text"] for chk in chunks]
    ids = [chk["chunk_id"] for chk in chunks]
    metadatas = [
        {
            "doc_id": chk["doc_id"],
            "title": chk.get("title", ""),
            "crop": chk.get("crop", "general"),
            "category": chk.get("category", "general"),
            "source_url": chk.get("source_url", ""),
            "license": chk.get("license", "")
        }
        for chk in chunks
    ]

    embeddings_list = None
    use_offline = os.environ.get("OFFLINE_MODE") == "1" or os.environ.get("LLM_PROVIDER") == "mock"

    if not use_offline:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Computing dense embeddings using {embedding_model_name}...")
            model = SentenceTransformer(embedding_model_name)
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            embeddings_list = embeddings.tolist()
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to deterministic dense vectors.")

    if embeddings_list is None:
        logger.info("Generating deterministic dense vectors for index (offline mode)...")
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm='l2')
        embeddings_list = vectorizer.transform(texts).toarray().tolist()
        embeddings_list = vectorizer.transform(texts).toarray().tolist()

    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings_list[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )

    logger.info(f"Persisted {len(chunks)} chunks into Chroma collection '{collection_name}' at {db_path}")
    return collection


def build_index(
    corpus_file: str = "data/corpus/chunks.jsonl",
    index_dir: str = "data/index"
):
    """Main indexing workflow: loads corpus chunks, saves metadata, builds BM25 and Chroma."""
    idx_path = Path(index_dir)
    idx_path.mkdir(parents=True, exist_ok=True)

    if not Path(corpus_file).exists():
        logger.info(f"Corpus file {corpus_file} not found. Running build_corpus.py first...")
        from scripts.build_corpus import build_corpus
        build_corpus()

    chunks = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {corpus_file}")

    # Save chunks metadata mapping for quick retrieval
    meta_path = idx_path / "chunks_metadata.json"
    meta_dict = {chk["chunk_id"]: chk for chk in chunks}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    logger.info(f"Saved metadata mapping to {meta_path}")

    # Build BM25
    bm25_file = idx_path / "bm25.pkl"
    build_bm25_index(chunks, output_path=str(bm25_file))

    # Build Chroma
    chroma_dir = idx_path / "chroma"
    build_chroma_collection(chunks, db_dir=str(chroma_dir))

    logger.info("Indexing completed successfully!")


if __name__ == "__main__":
    build_index()
