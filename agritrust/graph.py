"""LangGraph StateGraph implementation for AgriTrust.
Integrates Query Understanding, Routing, Hybrid Retrieval, Specialized Agents,
Synthesizer, Trust Layer (Verifier, Agreement, Confidence, Controller), and Responder.
"""

from typing import Dict, Any, List, Optional
import logging
import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END

from agritrust.state import AgriTrustState
from agritrust.llm import LLMClient
from agritrust.retrieval.hybrid import HybridRetriever
from agritrust.agents.crop_advisory import CropAdvisoryAgent
from agritrust.agents.plant_disease import PlantDiseaseAgent
from agritrust.agents.weather_context import WeatherContextAgent
from agritrust.agents.soil_fertilizer import SoilFertilizerAgent
from agritrust.nodes.query_understanding import query_understanding_node
from agritrust.nodes.router import router_node
from agritrust.nodes.synthesizer import synthesizer_node
from agritrust.nodes.responder import responder_node
from agritrust.trust.verifier import EvidenceVerifier
from agritrust.trust.agreement import AgreementAnalyzer
from agritrust.trust.confidence import ConfidenceEstimator
from agritrust.trust.controller import ResponseController

logger = logging.getLogger("agritrust.graph")


class AgriTrustPipeline:
    """End-to-end AgriTrust advisory pipeline powered by LangGraph."""

    def __init__(self, config_path: str = "agritrust/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize shared components
        self.llm_client = LLMClient(config_path=config_path)
        self.retriever = HybridRetriever(config_path=config_path)

        # Initialize agents
        self.agents = {
            "CropAdvisoryAgent": CropAdvisoryAgent(llm_client=self.llm_client),
            "PlantDiseaseAgent": PlantDiseaseAgent(llm_client=self.llm_client),
            "WeatherContextAgent": WeatherContextAgent(llm_client=self.llm_client),
            "SoilFertilizerAgent": SoilFertilizerAgent(llm_client=self.llm_client),
        }

        # Initialize trust layer
        trust_cfg = self.config.get("trust", {})
        weights = trust_cfg.get("weights", {})
        self.verifier = EvidenceVerifier(
            model_name=self.config.get("nli_verifier", {}).get("model_name", "cross-encoder/nli-deberta-v3-small")
        )
        self.agreement_analyzer = AgreementAnalyzer()
        self.confidence_estimator = ConfidenceEstimator(
            w1_ess=weights.get("w1_ess", 0.40),
            w2_agr=weights.get("w2_agr", 0.25),
            w3_retrieval=weights.get("w3_retrieval", 0.20),
            w4_self_conf=weights.get("w4_self_conf", 0.15),
            contradiction_penalty=trust_cfg.get("contradiction_penalty", 0.35),
        )
        self.controller = ResponseController(
            tau_high=trust_cfg.get("tau_high", 0.70),
            tau_low=trust_cfg.get("tau_low", 0.45),
            min_evidence_coverage=trust_cfg.get("min_evidence_coverage", 0.60),
            max_retries=trust_cfg.get("max_retrieval_retries", 2),
        )

        self.graph = self._build_graph()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # Node Functions
    def _node_query_understanding(self, state: AgriTrustState) -> Dict[str, Any]:
        return query_understanding_node(state, llm_client=self.llm_client)

    def _node_router(self, state: AgriTrustState) -> Dict[str, Any]:
        return router_node(state)

    def _node_retriever(self, state: AgriTrustState) -> Dict[str, Any]:
        query = state.get("query", "")
        context = state.get("extracted_entities", {})
        attempt = state.get("retrieval_attempt", 0)
        selected_agents = state.get("selected_agents", ["CropAdvisoryAgent"])

        # Loop counter
        current_attempt = attempt + 1
        k = self.retriever.expanded_top_k if current_attempt > 1 else self.retriever.top_k

        # Per-agent query rewriting
        agent_queries = {}
        for agent_name in selected_agents:
            agent_queries[agent_name] = self.retriever.rewrite_query_for_agent(query, agent_name, context)

        # Retrieve chunks for the primary intent / expanded queries
        primary_agent = selected_agents[0] if selected_agents else "CropAdvisoryAgent"
        target_query = agent_queries.get(primary_agent, query)

        chunks = self.retriever.retrieve_hybrid(target_query, top_k=k)

        return {
            "retrieval_attempt": current_attempt,
            "max_retrieval_retries": self.controller.max_retries,
            "retrieval_queries": agent_queries,
            "retrieved_chunks": chunks,
        }

    def _node_agents(self, state: AgriTrustState) -> Dict[str, Any]:
        query = state.get("query", "")
        chunks = state.get("retrieved_chunks", [])
        context = state.get("extracted_entities", {})
        selected = state.get("selected_agents", ["CropAdvisoryAgent"])

        agent_outputs = {}
        for agent_name in selected:
            agent = self.agents.get(agent_name)
            if agent:
                out = agent.run(query=query, retrieved_chunks=chunks, user_context=context)
                agent_outputs[agent_name] = out.model_dump()

        return {"agent_outputs": agent_outputs}

    def _node_synthesizer(self, state: AgriTrustState) -> Dict[str, Any]:
        return synthesizer_node(state, llm_client=self.llm_client)

    def _node_verifier(self, state: AgriTrustState) -> Dict[str, Any]:
        candidate_claims = state.get("candidate_claims", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        verifications, ess, coverage = self.verifier.verify_claims(candidate_claims, retrieved_chunks)
        return {
            "claim_verifications": verifications,
            "evidence_support_score": ess,
            "evidence_coverage": coverage,
        }

    def _node_agreement(self, state: AgriTrustState) -> Dict[str, Any]:
        agent_outputs = state.get("agent_outputs", {})
        agr, conflicts = self.agreement_analyzer.analyze_agreement(agent_outputs)
        return {
            "agent_agreement_score": agr,
            "agent_conflicts": conflicts,
        }

    def _node_confidence(self, state: AgriTrustState) -> Dict[str, Any]:
        ess = state.get("evidence_support_score", 0.0)
        agr = state.get("agent_agreement_score", 1.0)
        chunks = state.get("retrieved_chunks", [])
        outputs = state.get("agent_outputs", {})
        conflicts = state.get("agent_conflicts", [])
        verifications = state.get("claim_verifications", [])

        has_contra = any(v.get("label") == "contradicted" for v in verifications) or len(conflicts) > 0

        conf = self.confidence_estimator.estimate_confidence(
            evidence_support_score=ess,
            agent_agreement_score=agr,
            retrieved_chunks=chunks,
            agent_outputs=outputs,
            has_contradictions=has_contra,
        )
        return {"overall_confidence": conf}

    def _node_controller(self, state: AgriTrustState) -> Dict[str, Any]:
        is_ambiguous = state.get("is_ambiguous", False)
        conf = state.get("overall_confidence", 0.0)
        coverage = state.get("evidence_coverage", 0.0)
        conflicts = state.get("agent_conflicts", [])
        verifications = state.get("claim_verifications", [])
        attempt = state.get("retrieval_attempt", 1)

        has_contra = any(v.get("label") == "contradicted" for v in verifications) or len(conflicts) > 0

        decision, reasons = self.controller.decide(
            is_ambiguous=is_ambiguous,
            overall_confidence=conf,
            evidence_coverage=coverage,
            has_contradictions=has_contra,
            retrieval_attempt=attempt,
        )

        return {
            "decision": decision,
            "decision_reasons": reasons,
        }

    def _node_responder(self, state: AgriTrustState) -> Dict[str, Any]:
        return responder_node(state)

    def _route_controller(self, state: AgriTrustState) -> str:
        """Evaluate if controller requests RETRIEVE_MORE or final responder."""
        decision = state.get("decision", "ABSTAIN")
        attempt = state.get("retrieval_attempt", 1)
        max_retries = state.get("max_retrieval_retries", 2)

        if decision == "RETRIEVE_MORE" and attempt < max_retries:
            return "retriever"
        return "responder"

    def _build_graph(self):
        """Construct StateGraph with full multi-agent trust pipeline."""
        workflow = StateGraph(AgriTrustState)

        # Register nodes
        workflow.add_node("query_understanding", self._node_query_understanding)
        workflow.add_node("router", self._node_router)
        workflow.add_node("retriever", self._node_retriever)
        workflow.add_node("agents", self._node_agents)
        workflow.add_node("synthesizer", self._node_synthesizer)
        workflow.add_node("verifier", self._node_verifier)
        workflow.add_node("agreement_analyzer", self._node_agreement)
        workflow.add_node("confidence_estimator", self._node_confidence)
        workflow.add_node("controller", self._node_controller)
        workflow.add_node("responder", self._node_responder)

        # Normal edges
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "router")
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "agents")
        workflow.add_edge("agents", "synthesizer")
        workflow.add_edge("synthesizer", "verifier")
        workflow.add_edge("verifier", "agreement_analyzer")
        workflow.add_edge("agreement_analyzer", "confidence_estimator")
        workflow.add_edge("confidence_estimator", "controller")

        # Conditional edge from controller
        workflow.add_conditional_edges(
            "controller",
            self._route_controller,
            {
                "retriever": "retriever",
                "responder": "responder",
            }
        )

        workflow.add_edge("responder", END)

        return workflow.compile()

    def advise(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AgriTrustState:
        """Run end-to-end advisory on a query."""
        initial_state: AgriTrustState = {
            "query": query,
            "user_context": user_context or {},
            "retrieval_attempt": 0,
            "max_retrieval_retries": self.controller.max_retries,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state
