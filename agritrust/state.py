"""State definitions and schemas for the AgriTrust multi-agent framework."""

from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Structured output returned by each specialized agricultural agent."""
    recommendation: str = Field(..., description="Agent-specific domain recommendation.")
    key_claims: List[str] = Field(default_factory=list, description="Key atomic statements made by the agent.")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks supporting this output.")
    self_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Self-assessed confidence score [0, 1].")


class ClaimVerification(BaseModel):
    """Verification outcome for an individual atomic claim."""
    claim: str
    label: str  # 'supported' | 'unsupported' | 'contradicted'
    support_score: float  # [0, 1] entailment probability
    contradiction_score: float = 0.0
    best_chunk_id: Optional[str] = None
    best_chunk_text: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Retrieved evidence chunk with provenance metadata."""
    chunk_id: str
    doc_id: str
    source_url: str
    crop: Optional[str] = None
    title: Optional[str] = None
    text: str
    score: float  # Normalized relevance score [0, 1]
    rank: int = 1


class AgentConflict(BaseModel):
    """Conflict detected between two specialized agents."""
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    nli_contradiction_score: float
    description: str


class AgriTrustState(TypedDict, total=False):
    """Global TypedDict state flowing through the LangGraph StateGraph."""
    # Input & context
    query: str
    user_context: Dict[str, Any]

    # Query Understanding
    intent: str  # 'crop_advice' | 'disease' | 'weather_context' | 'soil_fertilizer' | 'general'
    extracted_entities: Dict[str, Any]
    is_ambiguous: bool
    clarifying_question: Optional[str]

    # Routing & Retrieval
    selected_agents: List[str]
    retrieval_queries: Dict[str, str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_attempt: int
    max_retrieval_retries: int

    # Agent Execution
    agent_outputs: Dict[str, Dict[str, Any]]

    # Synthesizer
    candidate_recommendation: str
    candidate_claims: List[str]
    candidate_citations: List[str]

    # Trust Layer
    claim_verifications: List[Dict[str, Any]]
    evidence_support_score: float  # ESS
    evidence_coverage: float      # fraction of supported claims
    agent_agreement_score: float   # AGR
    agent_conflicts: List[Dict[str, Any]]
    overall_confidence: float      # Conf

    # Response Control & Final Output
    decision: str  # 'ANSWER' | 'CLARIFY' | 'RETRIEVE_MORE' | 'ABSTAIN'
    decision_reasons: List[str]
    final_answer: str
    supported_claims: List[str]
    dropped_claims: List[str]
