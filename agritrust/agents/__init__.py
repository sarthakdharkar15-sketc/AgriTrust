"""Specialized agricultural agents package."""

from agritrust.agents.base import BaseAgriculturalAgent
from agritrust.agents.crop_advisory import CropAdvisoryAgent
from agritrust.agents.plant_disease import PlantDiseaseAgent
from agritrust.agents.weather_context import WeatherContextAgent
from agritrust.agents.soil_fertilizer import SoilFertilizerAgent

__all__ = [
    "BaseAgriculturalAgent",
    "CropAdvisoryAgent",
    "PlantDiseaseAgent",
    "WeatherContextAgent",
    "SoilFertilizerAgent",
]
