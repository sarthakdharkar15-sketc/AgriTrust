"""Baseline 2: Conventional RAG baseline with standard single-agent retrieval grounding."""

from typing import Dict, Any, Optional
from agritrust.llm import LLMClient
from agritrust.retrieval.hybrid import HybridRetriever


class ConventionalRAGBaseline:
    """Conventional single-query RAG without multi-agent decomposition or trust layer."""

    def __init__(self, config_path: str = "agritrust/config.yaml"):
        self.llm_client = LLMClient(config_path=config_path)
        self.retriever = HybridRetriever(config_path=config_path)

    def advise(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve chunks and generate answer without verification or confidence gating."""
        chunks = self.retriever.retrieve_hybrid(query, top_k=self.retriever.top_k)

        context_snippets = []
        for chk in chunks:
            cid = chk.get("chunk_id", "")
            txt = chk.get("text", "")
            context_snippets.append(f"[{cid}] {txt}")

        evidence_str = "\n\n".join(context_snippets)

        prompt = (
            f"You are an agricultural advisor. Answer the following user query based on the retrieved evidence:\n\n"
            f"Retrieved Evidence:\n{evidence_str}\n\n"
            f"User Query: {query}\n"
            f"User Context: {user_context or {}}\n\n"
            f"Provide an informative recommendation grounded in the retrieved documents."
        )

        answer = self.llm_client.complete(prompt)
        return {
            "query": query,
            "decision": "ANSWER",
            "final_answer": answer,
            "overall_confidence": 0.7,
            "retrieved_chunks": chunks,
            "supported_claims": [],
            "dropped_claims": [],
            "baseline": "conventional_rag",
        }
