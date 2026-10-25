"""Cost-safe, multi-provider LLM abstraction layer with disk caching, rate limiting, and retries."""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar, Any, Dict
from pydantic import BaseModel, ValidationError
import yaml

logger = logging.getLogger("agritrust.llm")

T = TypeVar("T", bound=BaseModel)


class DiskCache:
    """Thread-safe persistent disk cache for LLM responses."""

    def __init__(self, cache_dir: str = ".cache/llm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, prompt: str, model: str, temperature: float, extra: str = "") -> str:
        payload = f"{model}:{temperature:.2f}:{extra}:{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str, temperature: float, extra: str = "") -> Optional[str]:
        key = self._hash_key(prompt, model, temperature, extra)
        entry_path = self.cache_dir / f"{key}.json"
        if entry_path.exists():
            try:
                with open(entry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("response")
            except Exception as e:
                logger.warning(f"Failed to read cache key {key}: {e}")
        return None

    def set(self, prompt: str, model: str, temperature: float, response: str, extra: str = "") -> None:
        key = self._hash_key(prompt, model, temperature, extra)
        entry_path = self.cache_dir / f"{key}.json"
        try:
            with open(entry_path, "w", encoding="utf-8") as f:
                json.dump({"prompt": prompt, "model": model, "temperature": temperature, "response": response}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache key {key}: {e}")


class RateLimiter:
    """Simple rate limiter to respect requests per second limits."""

    def __init__(self, requests_per_second: float = 2.0):
        self.interval = 1.0 / max(requests_per_second, 0.1)
        self.last_call = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


class MockLLMHandler:
    """Deterministic mock provider producing compliant responses for zero-cost execution and tests."""

    @staticmethod
    def generate_response(prompt: str, schema: Optional[Type[BaseModel]] = None) -> str:
        prompt_lower = prompt.lower()

        # Handle Query Understanding Schema
        if "query_understanding" in prompt_lower or "classify intent" in prompt_lower or (schema and "intent" in schema.model_fields):
            intent = "general"
            if any(w in prompt_lower for w in ["disease", "blight", "rust", "yellowing", "spot", "wilt", "pest"]):
                intent = "disease"
            elif any(w in prompt_lower for w in ["fertilizer", "urea", "npk", "soil", "nutrient", "nitrogen"]):
                intent = "soil_fertilizer"
            elif any(w in prompt_lower for w in ["weather", "rain", "drought", "temperature", "frost", "monsoon"]):
                intent = "weather_context"
            elif any(w in prompt_lower for w in ["wheat", "rice", "sow", "variety", "yield", "cotton", "soybean", "maize"]):
                intent = "crop_advice"

            is_ambiguous = "ambiguous" in prompt_lower or "missing info" in prompt_lower or (
                "how much to apply" in prompt_lower and not any(c in prompt_lower for c in ["wheat", "rice", "cotton", "soybean", "maize", "tomato", "potato", "sugarcane"])
            )
            clarifying_question = "Could you please specify your crop, growth stage, and current soil conditions?" if is_ambiguous else None

            # Entity extraction
            crop = None
            for c in ["wheat", "rice", "soybean", "cotton", "maize", "tomato", "potato", "sugarcane"]:
                if c in prompt_lower:
                    crop = c
                    break

            result = {
                "intent": intent,
                "extracted_entities": {
                    "crop": crop,
                    "region": "Northern Plains" if "punjab" in prompt_lower or "haryana" in prompt_lower else None,
                    "season": "rabi" if "rabi" in prompt_lower or "winter" in prompt_lower else ("kharif" if "kharif" in prompt_lower or "summer" in prompt_lower else None),
                    "growth_stage": "vegetative" if "vegetative" in prompt_lower else None,
                    "symptoms": ["yellow leaves"] if "yellow" in prompt_lower else [],
                    "soil": "alluvial" if "alluvial" in prompt_lower else None,
                },
                "is_ambiguous": is_ambiguous,
                "clarifying_question": clarifying_question,
            }
            return json.dumps(result)

        # Handle Agent Output Schema
        if schema and "recommendation" in schema.model_fields:
            agent_type = "Advisory"
            if "disease" in prompt_lower:
                agent_type = "Plant Disease"
                rec = "Apply recommended systemic fungicide (e.g., Propiconazole at 1ml/L) and ensure field drainage to curtail fungal spread."
                claims = [
                    "Propiconazole at 1ml/L controls fungal foliar blights.",
                    "Proper field aeration and drainage reduces disease pressure."
                ]
            elif "soil" in prompt_lower or "fertilizer" in prompt_lower:
                agent_type = "Soil & Fertilizer"
                rec = "Apply balanced NPK at 120:60:40 kg/ha with basal phosphorus and split nitrogen application."
                claims = [
                    "Optimal fertilizer dosage is NPK 120:60:40 kg/ha.",
                    "Splitting nitrogen between basal and crown root stages optimizes nutrient uptake."
                ]
            elif "weather" in prompt_lower:
                agent_type = "Weather Context"
                rec = "Delay chemical spraying if heavy rainfall is expected within 24 hours to prevent runoff."
                claims = [
                    "Chemical sprays require a rain-free window of 24 hours.",
                    "High humidity favors foliar fungal germination."
                ]
            else:
                rec = "Maintain optimal seed rate of 100 kg/ha and ensure timely irrigation at critical growth stages."
                claims = [
                    "Recommended seed rate is 100 kg/ha for timely sown irrigated conditions.",
                    "First irrigation is critical at the crown root initiation stage."
                ]

            result = {
                "recommendation": f"[{agent_type}] {rec}",
                "key_claims": claims,
                "cited_chunk_ids": ["chunk_wheat_001", "chunk_fertilizer_002"],
                "self_confidence": 0.88,
            }
            return json.dumps(result)

        # Handle Synthesizer Schema
        if (schema and "candidate_recommendation" in schema.model_fields) or "synthesizer" in prompt_lower or "merge agent outputs" in prompt_lower:
            return json.dumps({
                "candidate_recommendation": "Maintain optimal seed rate of 100 kg/ha and apply balanced NPK 120:60:40 kg/ha. Avoid spraying if rainfall is forecasted within 24 hours.",
                "candidate_claims": [
                    "Recommended seed rate is 100 kg/ha for timely sown irrigated conditions.",
                    "Optimal fertilizer dosage is NPK 120:60:40 kg/ha.",
                    "Avoid chemical spraying if rainfall is expected within 24 hours."
                ],
                "candidate_citations": ["chunk_wheat_001", "chunk_fertilizer_002"]
            })

        # General text answer
        if "unanswerable" in prompt_lower or "not in corpus" in prompt_lower:
            return "I do not have sufficient validated agronomic evidence in the knowledge base to answer this query."
        return "Based on agronomic evidence, follow standard extension recommendations for variety selection, balanced fertilization, and moisture management."


class LLMClient:
    """Cost-safe LLM client supporting Gemini, Ollama, and Mock providers with caching and retries."""

    def __init__(self, config_path: str = "agritrust/config.yaml"):
        self.config = self._load_config(config_path)
        self.provider = os.getenv("LLM_PROVIDER", self.config.get("llm", {}).get("provider", "mock")).lower()
        self.gemini_model = os.getenv("GEMINI_MODEL", self.config.get("llm", {}).get("gemini_model", "gemini-1.5-flash"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.config.get("llm", {}).get("ollama_model", "llama3.2"))
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.config.get("llm", {}).get("ollama_base_url", "http://localhost:11434"))
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.2))
        self.max_retries = int(self.config.get("llm", {}).get("max_retries", 3))
        self.backoff_factor = float(self.config.get("llm", {}).get("backoff_factor", 1.5))

        cache_dir = os.getenv("CACHE_DIR", self.config.get("llm", {}).get("cache_dir", ".cache/llm"))
        self.cache = DiskCache(cache_dir)
        rate_limit = float(self.config.get("llm", {}).get("rate_limit_rps", 2.0))
        self.rate_limiter = RateLimiter(rate_limit)

        self._gemini_configured = False
        if self.provider == "gemini":
            self._init_gemini()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_gemini(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._gemini_configured = True
            except Exception as e:
                logger.warning(f"Could not configure google.generativeai: {e}. Falling back to mock.")
                self.provider = "mock"
        else:
            logger.warning("GEMINI_API_KEY not found in environment. Falling back to mock provider.")
            self.provider = "mock"

    def complete(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Execute a text completion with caching and rate limiting."""
        temp = temperature if temperature is not None else self.temperature
        model_name = self.gemini_model if self.provider == "gemini" else (self.ollama_model if self.provider == "ollama" else "mock")

        # Check Cache
        cached = self.cache.get(prompt, model_name, temp, extra=system_prompt or "")
        if cached is not None:
            return cached

        # Enforce Rate Limit for real API providers
        if self.provider != "mock":
            self.rate_limiter.wait()

        # Call Provider with retries
        response_text = ""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "gemini" and self._gemini_configured:
                    import google.generativeai as genai
                    model = genai.GenerativeModel(
                        model_name=self.gemini_model,
                        system_instruction=system_prompt
                    )
                    res = model.generate_content(
                        prompt,
                        generation_config={"temperature": temp}
                    )
                    response_text = res.text or ""
                elif self.provider == "ollama":
                    import requests
                    payload = {
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt or "",
                        "stream": False,
                        "options": {"temperature": temp}
                    }
                    resp = requests.post(f"{self.ollama_base_url}/api/generate", json=payload, timeout=60)
                    resp.raise_for_status()
                    response_text = resp.json().get("response", "")
                else:  # Mock
                    response_text = MockLLMHandler.generate_response(prompt)

                # Store in cache
                if response_text:
                    self.cache.set(prompt, model_name, temp, response_text, extra=system_prompt or "")
                    return response_text

            except Exception as e:
                last_error = e
                sleep_time = self.backoff_factor ** attempt
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        # Fallback to mock on total failure to guarantee graceful degradation
        logger.error(f"All LLM retries failed ({last_error}). Falling back to deterministic mock response.")
        fallback = MockLLMHandler.generate_response(prompt)
        self.cache.set(prompt, model_name, temp, fallback, extra=system_prompt or "")
        return fallback

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> T:
        """Generate structured output adhering to a Pydantic schema with validation and retry."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"You MUST respond ONLY with a valid JSON object matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any other markdown formatting or conversational commentary outside the JSON block."
        )

        for attempt in range(2):
            raw_text = self.complete(structured_prompt, system_prompt=system_prompt, temperature=temperature)
            cleaned = self._extract_json(raw_text)
            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Structured output parsing attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    structured_prompt += f"\n\nERROR IN PREVIOUS ATTEMPT: {e}. Fix JSON formatting and ensure strict schema compliance."

        # Deterministic fallback on complete parse failure
        logger.error(f"Failed to produce valid structured output for {schema.__name__}. Using deterministic mock fallback.")
        mock_raw = MockLLMHandler.generate_response(prompt, schema=schema)
        mock_clean = self._extract_json(mock_raw)
        try:
            return schema.model_validate(json.loads(mock_clean))
        except Exception:
            # Construct empty or minimal valid model instance
            fields = {k: None for k in schema.model_fields.keys()}
            return schema.model_construct(**fields)

    def _extract_json(self, text: str) -> str:
        """Extract JSON substring from potential markdown fences."""
        text = text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
            return text[start:].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
            return text[start:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text
