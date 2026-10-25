"""Baseline 3: Multi-Agent RAG without trust layer (no verifier, agreement analyzer, or confidence control)."""

from typing import Dict, Any, Optional
from agritrust.llm import LLMClient
from agritrust.retrieval.hybrid import HybridRetriever
from agritrust.agents.crop_advisory import CropAdvisoryAgent
from agritrust.agents.plant_disease import PlantDiseaseAgent
from agritrust.agents.weather_context import WeatherContextAgent
from agritrust.agents.soil_fertilizer import SoilFertilizerAgent
from agritrust.nodes.synthesizer import synthesizer_node
from agritrust.nodes.router import router_node
from agritrust.nodes.query_understanding import query_understanding_node
from agritrust.state import AgriTrustState


class MultiAgentRAGBaseline:
    """Multi-Agent RAG without trust layer (no verification, conflict detection, or confidence gating)."""

    def __init__(self, config_path: str = "agritrust/config.yaml"):
        self.llm_client = LLMClient(config_path=config_path)
        self.retriever = HybridRetriever(config_path=config_path)
        self.agents = {
            "CropAdvisoryAgent": CropAdvisoryAgent(llm_client=self.llm_client),
            "PlantDiseaseAgent": PlantDiseaseAgent(llm_client=self.llm_client),
            "WeatherContextAgent": WeatherContextAgent(llm_client=self.llm_client),
            "SoilFertilizerAgent": SoilFertilizerAgent(llm_client=self.llm_client),
        }

    def advise(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run multi-agent retrieval and synthesis, but output recommendation directly without verification."""
        state: AgriTrustState = {
            "query": query,
            "user_context": user_context or {},
        }
        # 1. Query understanding
        qu_res = query_understanding_node(state, llm_client=self.llm_client)
        state.update(qu_res)

        # 2. Router
        router_res = router_node(state)
        state.update(router_res)

        # 3. Retrieve
        chunks = self.retriever.retrieve_hybrid(query, top_k=self.retriever.top_k)
        state["retrieved_chunks"] = chunks

        # 4. Agent execution
        agent_outputs = {}
        for agent_name in state.get("selected_agents", ["CropAdvisoryAgent"]):
            agent = self.agents.get(agent_name)
            if agent:
                out = agent.run(query=query, retrieved_chunks=chunks, user_context=state.get("extracted_entities"))
                agent_outputs[agent_name] = out.model_dump()
        state["agent_outputs"] = agent_outputs

        # 5. Synthesizer
        syn_res = synthesizer_node(state, llm_client=self.llm_client)
        state.update(syn_res)

        # Output directly (no trust layer, no verification, no response controller)
        return {
            "query": query,
            "decision": "ANSWER",  # MARAG without trust layer unconditionally answers
            "final_answer": state.get("candidate_recommendation", ""),
            "overall_confidence": 0.8,
            "retrieved_chunks": chunks,
            "supported_claims": state.get("candidate_claims", []),
            "dropped_claims": [],
            "baseline": "marag_no_trust",
        }
