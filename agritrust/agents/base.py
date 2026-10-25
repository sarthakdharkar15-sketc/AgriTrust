"""Base class and utilities for specialized agricultural agents."""

from typing import List, Dict, Any, Optional
import logging
from agritrust.state import AgentOutput
from agritrust.llm import LLMClient

logger = logging.getLogger("agritrust.agents")


class BaseAgriculturalAgent:
    """Base class for domain-specialized agricultural advisory agents."""

    def __init__(self, name: str, domain: str, llm_client: Optional[LLMClient] = None):
        self.name = name
        self.domain = domain
        self.llm_client = llm_client or LLMClient()

    def _format_evidence(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks with chunk IDs for citation grounding."""
        if not chunks:
            return "No specific evidence chunks retrieved."

        lines = []
        for chk in chunks:
            cid = chk.get("chunk_id", "unknown")
            crop = chk.get("crop", "general")
            title = chk.get("title", "")
            text = chk.get("text", "").strip()
            lines.append(f"[{cid}] (Crop: {crop} | Title: {title})\n{text}")
        return "\n\n".join(lines)

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        raise NotImplementedError
