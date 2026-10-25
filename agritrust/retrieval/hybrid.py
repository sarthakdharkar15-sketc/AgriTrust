"""Hybrid Retriever combining BM25 and Dense MiniLM retrieval with Reciprocal Rank Fusion."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("agritrust.retrieval")


class HybridRetriever:
    """Hybrid BM25 + Dense vector retriever with RRF and per-agent query rewriting."""

    def __init__(
        self,
        config_path: str = "agritrust/config.yaml",
        bm25_index_path: Optional[str] = None,
        chroma_db_dir: Optional[str] = None,
        metadata_index_path: Optional[str] = None,
    ):
        import yaml
        self.config = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        ret_conf = self.config.get("retrieval", {})
        self.top_k = ret_conf.get("top_k", 5)
        self.expanded_top_k = ret_conf.get("expanded_top_k", 8)
        self.bm25_weight = ret_conf.get("bm25_weight", 0.5)
        self.dense_weight = ret_conf.get("dense_weight", 0.5)
        self.rrf_k = ret_conf.get("rrf_k", 60)

        self.bm25_path = Path(bm25_index_path or ret_conf.get("bm25_index_path", "data/index/bm25.pkl"))
        self.chroma_dir = Path(chroma_db_dir or ret_conf.get("chroma_db_dir", "data/index/chroma"))
        self.meta_path = Path(metadata_index_path or ret_conf.get("metadata_index_path", "data/index/chunks_metadata.json"))

        self.bm25 = None
        self.metadata = {}
        self.chunk_ids = []
        self.chroma_collection = None
        self.embedding_model = None

        self._load_indices()

    def _load_indices(self) -> None:
        """Load persisted BM25 index, metadata mapping, and Chroma collection."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.chunk_ids = list(self.metadata.keys())
            except Exception as e:
                logger.warning(f"Failed to load metadata {self.meta_path}: {e}")

        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load BM25 index {self.bm25_path}: {e}")

        if self.chroma_dir.exists():
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                self.chroma_collection = client.get_collection("agritrust_corpus")
            except Exception as e:
                logger.warning(f"Chroma collection not loaded: {e}")

        self.vectorizer = None
        use_offline = os.environ.get("OFFLINE_MODE") == "1" or os.environ.get("LLM_PROVIDER") == "mock"
        if not use_offline:
            try:
                from sentence_transformers import SentenceTransformer
                emb_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(emb_name)
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None

        if self.embedding_model is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm='l2')

    def rewrite_query_for_agent(self, base_query: str, agent_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Domain-specific query rewriting per specialized agent."""
        ctx_crop = (context or {}).get("crop", "")
        crop_prefix = f"{ctx_crop} " if ctx_crop and ctx_crop.lower() not in base_query.lower() else ""

        if "disease" in agent_name.lower():
            return f"{crop_prefix}{base_query} plant disease fungal bacterial pathogen symptoms fungicide treatment spray dosage"
        elif "fertilizer" in agent_name.lower() or "soil" in agent_name.lower():
            return f"{crop_prefix}{base_query} fertilizer NPK nutrient dosage basal top dressing deficiency zinc sulfur"
        elif "weather" in agent_name.lower():
            return f"{crop_prefix}{base_query} weather climate rain temperature frost moisture humidity spray conditions"
        elif "crop" in agent_name.lower():
            return f"{crop_prefix}{base_query} sowing seed rate variety irrigation spacing harvest yield agronomy"
        return f"{crop_prefix}{base_query}"

    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top chunks via BM25."""
        if not self.bm25 or not self.chunk_ids:
            return []

        tokens = [w.strip(".,;:?!()[]'\"").lower() for w in query.split() if w.strip()]
        scores = self.bm25.get_scores(tokens)

        # Top indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0 and len(results) >= 3:
                break
            cid = self.chunk_ids[idx]
            meta = self.metadata.get(cid, {})
            results.append({
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", "unknown"),
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
                "crop": meta.get("crop", "general"),
                "text": meta.get("text", ""),
                "bm25_score": score,
            })
        return results

    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top chunks via Chroma vector search."""
        if not self.chroma_collection:
            return []

        try:
            if self.embedding_model is not None:
                query_vec = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
                chroma_res = self.chroma_collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(top_k, self.chroma_collection.count())
                )
            elif self.vectorizer is not None:
                query_vec = self.vectorizer.transform([query]).toarray().tolist()[0]
                chroma_res = self.chroma_collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(top_k, self.chroma_collection.count())
                )
            else:
                chroma_res = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.chroma_collection.count())
                )

            results = []
            if chroma_res and "ids" in chroma_res and chroma_res["ids"]:
                ids = chroma_res["ids"][0]
                distances = chroma_res.get("distances", [[]])[0]
                documents = chroma_res.get("documents", [[]])[0]
                metas = chroma_res.get("metadatas", [[]])[0]

                for i, cid in enumerate(ids):
                    dist = distances[i] if i < len(distances) else 0.5
                    # Cosine distance in Chroma: dist in [0, 2], similarity = 1 - dist
                    sim = max(0.0, 1.0 - float(dist))
                    meta = metas[i] if i < len(metas) else {}
                    results.append({
                        "chunk_id": cid,
                        "doc_id": meta.get("doc_id", "unknown"),
                        "source_url": meta.get("source_url", ""),
                        "title": meta.get("title", ""),
                        "crop": meta.get("crop", "general"),
                        "text": documents[i] if i < len(documents) else "",
                        "dense_similarity": sim,
                    })
            return results
        except Exception as e:
            logger.warning(f"Chroma dense retrieval failed: {e}")
            return []

    def retrieve_hybrid(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute hybrid search using Reciprocal Rank Fusion (RRF)."""
        k = top_k or self.top_k
        bm25_res = self.retrieve_bm25(query, top_k=k * 2)
        dense_res = self.retrieve_dense(query, top_k=k * 2)

        if not bm25_res and not dense_res:
            # Fallback if indices not yet built: use metadata directly
            fallback_items = list(self.metadata.values())[:k]
            return [
                {
                    "chunk_id": item.get("chunk_id", f"c_{idx}"),
                    "doc_id": item.get("doc_id", "fallback"),
                    "source_url": item.get("source_url", ""),
                    "crop": item.get("crop", "general"),
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "score": 0.5,
                    "rank": idx + 1
                }
                for idx, item in enumerate(fallback_items)
            ]

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        item_data: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(bm25_res):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
            item_data[cid] = item

        for rank, item in enumerate(dense_res):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            if cid not in item_data:
                item_data[cid] = item

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        min_rrf = min(rrf_scores.values()) if rrf_scores else 0.0
        range_rrf = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

        ranked_results = []
        for rank, cid in enumerate(sorted_cids, 1):
            raw_rrf = rrf_scores[cid]
            norm_score = float((raw_rrf - min_rrf) / range_rrf) if range_rrf > 0 else 0.8
            # Bound normalized score between 0.35 and 0.98 for realistic calibrated distribution
            final_score = round(0.35 + 0.63 * norm_score, 4)

            entry = item_data[cid]
            ranked_results.append({
                "chunk_id": cid,
                "doc_id": entry.get("doc_id", "unknown"),
                "source_url": entry.get("source_url", ""),
                "title": entry.get("title", ""),
                "crop": entry.get("crop", "general"),
                "text": entry.get("text", ""),
                "score": final_score,
                "rank": rank,
            })

        return ranked_results
