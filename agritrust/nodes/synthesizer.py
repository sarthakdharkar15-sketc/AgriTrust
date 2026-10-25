"""Synthesizer node: merge specialized agent recommendations into a unified candidate response."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from agritrust.state import AgriTrustState
from agritrust.llm import LLMClient


class SynthesisOutput(BaseModel):
    """Pydantic schema for merged candidate advisory."""
    candidate_recommendation: str = Field(..., description="Unified synthesis of agent recommendations.")
    candidate_claims: List[str] = Field(..., description="Discrete atomic claims making up the recommendation.")
    candidate_citations: List[str] = Field(default_factory=list, description="All chunk IDs cited across agent inputs.")


def synthesizer_node(state: AgriTrustState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Merge specialized agent recommendations into candidate claims and recommendation."""
    llm = llm_client or LLMClient()
    agent_outputs = state.get("agent_outputs", {})
    query = state.get("query", "")

    if not agent_outputs:
        return {
            "candidate_recommendation": "No specific agent advice was generated.",
            "candidate_claims": [],
            "candidate_citations": [],
        }

    # Format agent outputs
    agent_texts = []
    all_citations = set()
    for agent_name, output in agent_outputs.items():
        rec = output.get("recommendation", "")
        claims = output.get("key_claims", [])
        citations = output.get("cited_chunk_ids", [])
        all_citations.update(citations)
        agent_texts.append(f"### {agent_name}:\nRecommendation: {rec}\nClaims:\n- " + "\n- ".join(claims))

    combined_agent_text = "\n\n".join(agent_texts)

    system_prompt = (
        "You are the Synthesizer module in AgriTrust. Synthesize individual specialized agent outputs "
        "into a coherent, actionable agricultural advisory recommendation. "
        "Deconstruct the synthesis into a clear list of discrete, verifiable ATOMIC CLAIMS. "
        "Preserve cited chunk IDs where applicable."
    )

    user_prompt = (
        f"Original User Query: {query}\n\n"
        f"Specialized Agent Findings:\n{combined_agent_text}\n\n"
        f"Produce a consolidated recommendation and extract the atomic claims."
    )

    result = llm.generate_structured(
        prompt=user_prompt,
        schema=SynthesisOutput,
        system_prompt=system_prompt
    )

    # Combine citations
    cits = result.candidate_citations if result.candidate_citations is not None else []
    final_citations = sorted(list(set(cits).union(all_citations)))

    return {
        "candidate_recommendation": result.candidate_recommendation,
        "candidate_claims": result.candidate_claims,
        "candidate_citations": final_citations,
    }
