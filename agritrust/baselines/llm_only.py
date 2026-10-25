"""Baseline 1: Plain LLM baseline without retrieval or multi-agent orchestration."""

from typing import Dict, Any, Optional
from agritrust.llm import LLMClient


class LLMOnlyBaseline:
    """Plain LLM baseline without retrieval grounding or trust layer."""

    def __init__(self, config_path: str = "agritrust/config.yaml"):
        self.llm_client = LLMClient(config_path=config_path)

    def advise(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate answer using only parametric memory of the LLM."""
        prompt = (
            f"You are an agricultural advisor.\n\n"
            f"User Query: {query}\n"
            f"User Context: {user_context or {}}\n\n"
            f"Provide practical agricultural recommendations."
        )

        answer = self.llm_client.complete(prompt)
        return {
            "query": query,
            "decision": "ANSWER",
            "final_answer": answer,
            "overall_confidence": 0.5,
            "retrieved_chunks": [],
            "supported_claims": [],
            "dropped_claims": [],
            "baseline": "plain_llm",
        }
