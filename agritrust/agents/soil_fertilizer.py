"""Soil and Fertilizer Agent specializing in crop nutrition, NPK dosage, and soil health."""

from typing import List, Dict, Any, Optional
from agritrust.agents.base import BaseAgriculturalAgent
from agritrust.state import AgentOutput


class SoilFertilizerAgent(BaseAgriculturalAgent):
    """Specialized in soil fertility, NPK recommendations, split applications, and micronutrients."""

    def __init__(self, llm_client=None):
        super().__init__(name="SoilFertilizerAgent", domain="Soil Science & Nutrient Management", llm_client=llm_client)

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        evidence_text = self._format_evidence(retrieved_chunks)
        ctx_text = str(user_context or {})

        system_prompt = (
            "You are the Soil and Fertilizer Agent in AgriTrust, an expert in soil health, NPK ratios, "
            "micronutrients (zinc, sulfur, boron, iron), fertilizer timing (basal vs split), and soil amendments. "
            "Base recommendations strictly on retrieved evidence. Specify exact dosages and cite chunk IDs."
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Context: {ctx_text}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"Synthesize an expert soil and fertilizer recommendation. "
            f"List key claims, cited chunk IDs, and your self confidence."
        )

        return self.llm_client.generate_structured(
            prompt=user_prompt,
            schema=AgentOutput,
            system_prompt=system_prompt
        )
