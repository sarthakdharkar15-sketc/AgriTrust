"""Plant Disease Agent specializing in symptoms diagnosis, pathogens, and pest/disease management."""

from typing import List, Dict, Any, Optional
from agritrust.agents.base import BaseAgriculturalAgent
from agritrust.state import AgentOutput


class PlantDiseaseAgent(BaseAgriculturalAgent):
    """Specialized in crop pathology, fungal/bacterial/viral disease symptoms, and plant protection."""

    def __init__(self, llm_client=None):
        super().__init__(name="PlantDiseaseAgent", domain="Plant Pathology & Crop Protection", llm_client=llm_client)

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        evidence_text = self._format_evidence(retrieved_chunks)
        ctx_text = str(user_context or {})

        system_prompt = (
            "You are the Plant Disease Agent in AgriTrust, an expert in plant pathology, identifying symptoms, "
            "fungal/bacterial/viral etiologies, spray dosages, and integrated pest management. "
            "Base recommendations strictly on retrieved evidence. State dosages clearly and cite chunk IDs."
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Context: {ctx_text}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"Synthesize an expert plant pathology recommendation. Detail chemical/biological controls and dosage. "
            f"Extract discrete atomic key claims and cite chunk IDs."
        )

        return self.llm_client.generate_structured(
            prompt=user_prompt,
            schema=AgentOutput,
            system_prompt=system_prompt
        )
