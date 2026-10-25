"""Router node: select appropriate specialized agents based on intent and query."""

from typing import Dict, Any, List
from agritrust.state import AgriTrustState


def router_node(state: AgriTrustState) -> Dict[str, Any]:
    """Route query to relevant specialized agricultural agents."""
    intent = state.get("intent", "general")
    query = state.get("query", "").lower()

    agents = set()

    if intent == "crop_advice":
        agents.add("CropAdvisoryAgent")
        if any(w in query for w in ["fertilizer", "nutrient", "urea", "npk", "yield", "soil"]):
            agents.add("SoilFertilizerAgent")
        if any(w in query for w in ["rain", "weather", "temperature", "climate", "frost"]):
            agents.add("WeatherContextAgent")
    elif intent == "disease":
        agents.add("PlantDiseaseAgent")
        # Disease treatment often depends on weather spray window
        agents.add("WeatherContextAgent")
    elif intent == "weather_context":
        agents.add("WeatherContextAgent")
        agents.add("CropAdvisoryAgent")
    elif intent == "soil_fertilizer":
        agents.add("SoilFertilizerAgent")
        agents.add("CropAdvisoryAgent")
    else:  # general
        agents.add("CropAdvisoryAgent")
        agents.add("SoilFertilizerAgent")

    # If no agent selected, default to CropAdvisoryAgent
    selected_agents = sorted(list(agents)) if agents else ["CropAdvisoryAgent"]

    return {
        "selected_agents": selected_agents
    }
