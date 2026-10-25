"""Query understanding node: intent classification, entity extraction, ambiguity detection."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from agritrust.state import AgriTrustState
from agritrust.llm import LLMClient

INTENT_TYPES = ["crop_advice", "disease", "weather_context", "soil_fertilizer", "general"]


class QueryUnderstandingOutput(BaseModel):
    """Pydantic schema for query understanding."""
    intent: str = Field(..., description="Classified intent: crop_advice | disease | weather_context | soil_fertilizer | general")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities: crop, region, season, growth stage, symptoms, soil")
    is_ambiguous: bool = Field(default=False, description="Whether query lacks critical agricultural information")
    clarifying_question: Optional[str] = Field(default=None, description="Clarifying question if ambiguous")


def query_understanding_node(state: AgriTrustState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Analyze query intent, extract agricultural entities, and detect ambiguity."""
    llm = llm_client or LLMClient()
    query = state.get("query", "")
    user_context = state.get("user_context", {})

    system_prompt = (
        "You are the Query Understanding module for AgriTrust, an agricultural advisory system. "
        "Your task:\n"
        "1. Classify the intent into one of: 'crop_advice', 'disease', 'weather_context', 'soil_fertilizer', 'general'.\n"
        "2. Extract domain entities: crop, region, season, growth stage, symptoms, soil.\n"
        "3. Detect if critical information is missing (e.g. asking for pesticide or fertilizer dosage without specifying crop or symptoms). "
        "If critical info is missing, mark is_ambiguous=True and formulate a concise clarifying_question."
    )

    user_prompt = (
        f"User Query: {query}\n"
        f"Provided User Context: {user_context}\n\n"
        f"Perform intent classification, entity extraction, and ambiguity assessment."
    )

    result = llm.generate_structured(
        prompt=user_prompt,
        schema=QueryUnderstandingOutput,
        system_prompt=system_prompt
    )

    # Validate intent
    intent = result.intent if result.intent in INTENT_TYPES else "general"

    # Merge context with extracted entities
    entities = dict(result.extracted_entities)
    for k, v in user_context.items():
        if v and not entities.get(k):
            entities[k] = v

    return {
        "intent": intent,
        "extracted_entities": entities,
        "is_ambiguous": result.is_ambiguous,
        "clarifying_question": result.clarifying_question if result.is_ambiguous else None,
    }
