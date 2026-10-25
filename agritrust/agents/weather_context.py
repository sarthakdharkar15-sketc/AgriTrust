"""Weather Context Agent specializing in agrometeorology, rainfall windows, and climate risks."""

from typing import List, Dict, Any, Optional
from agritrust.agents.base import BaseAgriculturalAgent
from agritrust.state import AgentOutput


class WeatherContextAgent(BaseAgriculturalAgent):
    """Specialized in agrometeorological impacts, rain windows, temperature, and frost.
    Uses strictly user-provided/context text and retrieved documents (no external weather API).
    """

    def __init__(self, llm_client=None):
        super().__init__(name="WeatherContextAgent", domain="Agrometeorology & Climate Context", llm_client=llm_client)

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        evidence_text = self._format_evidence(retrieved_chunks)
        ctx_text = str(user_context or {})

        system_prompt = (
            "You are the Weather Context Agent in AgriTrust. Analyze agrometeorological considerations, "
            "spray suitability windows (rain-free periods, wind speeds), frost mitigation, drainage, and temperature impacts. "
            "IMPORTANT: Use ONLY user-provided context text and retrieved evidence documents. "
            "Do NOT call or assume external live weather APIs. Ground all claims in the evidence provided."
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Context: {ctx_text}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"Provide agrometeorological advice tailored to the query and evidence. "
            f"List key claims, cited chunk IDs, and self confidence."
        )

        return self.llm_client.generate_structured(
            prompt=user_prompt,
            schema=AgentOutput,
            system_prompt=system_prompt
        )
