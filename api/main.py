"""AgriTrust FastAPI Application.
Thin API wrapper exposing the AgriTrust pipeline via the /advise endpoint.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agritrust.graph import AgriTrustPipeline

app = FastAPI(
    title="AgriTrust API",
    description="Evidence- and Confidence-Aware Multi-Agent RAG Framework for Reliable Agricultural Advisory",
    version="0.1.0"
)

# Global pipeline instance
pipeline: Optional[AgriTrustPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = AgriTrustPipeline()


class AdviseRequest(BaseModel):
    query: str = Field(..., description="Agricultural question or user problem description.")
    crop: Optional[str] = Field(None, description="Target crop (e.g. wheat, rice, tomato, potato).")
    region: Optional[str] = Field(None, description="Agro-ecological region or state.")
    season: Optional[str] = Field(None, description="Season (e.g. kharif, rabi, zaid).")
    soil: Optional[str] = Field(None, description="Soil texture or classification.")
    growth_stage: Optional[str] = Field(None, description="Crop developmental stage.")


class AdviseResponse(BaseModel):
    query: str
    decision: str
    overall_confidence: float
    evidence_support_score: float
    evidence_coverage: float
    agent_agreement_score: float
    decision_reasons: List[str]
    final_answer: str
    supported_claims: List[str]
    dropped_claims: List[str]
    cited_chunk_ids: List[str]


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AgriTrust API", "version": "0.1.0"}


@app.post("/advise", response_model=AdviseResponse)
def advise(request: AdviseRequest):
    global pipeline
    if pipeline is None:
        pipeline = AgriTrustPipeline()

    user_context = {}
    if request.crop:
        user_context["crop"] = request.crop
    if request.region:
        user_context["region"] = request.region
    if request.season:
        user_context["season"] = request.season
    if request.soil:
        user_context["soil"] = request.soil
    if request.growth_stage:
        user_context["growth_stage"] = request.growth_stage

    try:
        res = pipeline.advise(request.query, user_context=user_context)

        # Collect cited chunk IDs
        cited_cids = []
        for v in res.get("claim_verifications", []):
            cid = v.get("best_chunk_id")
            if cid and cid not in cited_cids:
                cited_cids.append(cid)

        return AdviseResponse(
            query=request.query,
            decision=res.get("decision", "ABSTAIN"),
            overall_confidence=res.get("overall_confidence", 0.0),
            evidence_support_score=res.get("evidence_support_score", 0.0),
            evidence_coverage=res.get("evidence_coverage", 0.0),
            agent_agreement_score=res.get("agent_agreement_score", 1.0),
            decision_reasons=res.get("decision_reasons", []),
            final_answer=res.get("final_answer", ""),
            supported_claims=res.get("supported_claims", []),
            dropped_claims=res.get("dropped_claims", []),
            cited_chunk_ids=cited_cids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advisory pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
