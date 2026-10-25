"""Crop Advisory Agent specializing in variety selection, sowing, spacing, and agronomy."""

from typing import List, Dict, Any, Optional
from agritrust.agents.base import BaseAgriculturalAgent
from agritrust.state import AgentOutput


class CropAdvisoryAgent(BaseAgriculturalAgent):
    """Specialized in agronomic guidance, sowing windows, seed rates, irrigation, and varieties."""

    def __init__(self, llm_client=None):
        super().__init__(name="CropAdvisoryAgent", domain="Crop Agronomy & Management", llm_client=llm_client)

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        evidence_text = self._format_evidence(retrieved_chunks)
        ctx_text = str(user_context or {})

        system_prompt = (
            "You are the Crop Advisory Agent in AgriTrust, an expert in crop agronomy, sowing dates, "
            "seed rates, planting geometry, irrigation stages, and variety selection. "
            "Provide evidence-grounded recommendations citing specific chunk IDs like [chunk_id]. "
            "Extract discrete atomic key claims."
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Context: {ctx_text}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"Synthesize an expert crop agronomy recommendation strictly supported by the evidence above. "
            f"List key claims and chunk IDs you cited."
        )

        return self.llm_client.generate_structured(
            prompt=user_prompt,
            schema=AgentOutput,
            system_prompt=system_prompt
        )
