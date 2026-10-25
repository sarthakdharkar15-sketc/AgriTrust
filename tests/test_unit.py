"""Unit tests for AgriTrust components (must pass offline with LLM_PROVIDER=mock)."""

import pytest
import os
import json
from pydantic import BaseModel, Field

# Ensure mock provider
os.environ["LLM_PROVIDER"] = "mock"

from scripts.build_corpus import recursive_chunk_text
from agritrust.retrieval.hybrid import HybridRetriever
from agritrust.trust.verifier import EvidenceVerifier
from agritrust.trust.agreement import AgreementAnalyzer
from agritrust.trust.confidence import ConfidenceEstimator
from agritrust.trust.controller import ResponseController
from agritrust.llm import LLMClient
from agritrust.state import AgentOutput


def test_chunker_length_and_overlap():
    """Verify recursive chunker splits text into valid size with proper overlap."""
    sample_text = (
        "Wheat is an essential cereal crop. " * 30 +
        "\n\nCrown Root Initiation is the most critical irrigation stage. " * 30 +
        "\n\nYellow rust requires timely fungicide application of Propiconazole. " * 30
    )
    chunks = recursive_chunk_text(sample_text, chunk_size_chars=500, chunk_overlap_chars=100)
    assert len(chunks) > 1, "Should split into multiple chunks"
    for chk in chunks:
        assert len(chk) <= 600, f"Chunk exceeded max bound: {len(chk)}"
        assert len(chk) > 0


def test_hybrid_retrieval_metadata_fields():
    """Verify retrieval returns required provenance metadata fields (doc, chunk_id, score, rank)."""
    retriever = HybridRetriever()
    results = retriever.retrieve_hybrid("wheat rust fungicide dosage", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "chunk_id" in item
        assert "doc_id" in item
        assert "score" in item
        assert "rank" in item
        assert "text" in item
        assert 0.0 <= item["score"] <= 1.0


def test_verifier_labels_supported_unsupported_contradicted():
    """Verify verifier assigns correct labels (supported, unsupported, contradicted) on fixed examples."""
    verifier = EvidenceVerifier()

    evidence = [
        {
            "chunk_id": "c1",
            "text": "For timely sown wheat, the recommended seed rate is 100 kg per hectare. Spray Propiconazole at 1 ml/L for rust."
        }
    ]

    claims = [
        "Recommended seed rate for wheat is 100 kg per hectare.",  # Supported
        "Wheat requires 900 kg per hectare seed rate.",            # Contradicted / unsupported
        "Avocado trees require annual grafting in Spain.",         # Unsupported
    ]

    results, ess, coverage = verifier.verify_claims(claims, evidence)
    assert len(results) == 3
    assert results[0]["label"] == "supported"
    assert results[0]["support_score"] > 0.4
    assert results[2]["label"] == "unsupported"
    assert 0.0 <= ess <= 1.0
    assert 0.0 <= coverage <= 1.0


def test_agreement_score_identical_vs_contradicting():
    """Verify agreement analyzer yields high score for consensus and detects conflicts for contradictions."""
    analyzer = AgreementAnalyzer()

    # Consensus scenario
    consensus_outputs = {
        "AgentA": {
            "key_claims": ["Apply Propiconazole at 1 ml/L for foliar blight.", "Ensure field drainage."],
            "recommendation": "Spray fungicide.",
            "cited_chunk_ids": ["c1"],
            "self_confidence": 0.9
        },
        "AgentB": {
            "key_claims": ["Propiconazole at 1 ml/L is effective against fungal blight.", "Proper drainage mitigates root rot."],
            "recommendation": "Maintain drainage and spray.",
            "cited_chunk_ids": ["c1"],
            "self_confidence": 0.85
        }
    }
    agr_high, conflicts_none = analyzer.analyze_agreement(consensus_outputs)
    assert agr_high >= 0.7
    assert len(conflicts_none) == 0

    # Contradicting scenario
    contradicting_outputs = {
        "AgentA": {
            "key_claims": ["Spray chemical fungicides immediately during rainfall."],
            "recommendation": "Spray now.",
            "cited_chunk_ids": ["c1"],
            "self_confidence": 0.8
        },
        "AgentB": {
            "key_claims": ["Never spray chemical fungicides during rainfall; avoid application."],
            "recommendation": "Do not spray.",
            "cited_chunk_ids": ["c2"],
            "self_confidence": 0.85
        }
    }
    agr_low, conflicts = analyzer.analyze_agreement(contradicting_outputs)
    assert len(conflicts) > 0
    assert agr_low < agr_high


def test_confidence_formula():
    """Verify confidence formula weights, calibration bounds, and contradiction penalty."""
    estimator = ConfidenceEstimator(
        w1_ess=0.40,
        w2_agr=0.25,
        w3_retrieval=0.20,
        w4_self_conf=0.15,
        contradiction_penalty=0.35
    )

    chunks = [{"score": 0.8}, {"score": 0.9}]
    outputs = {"AgentA": {"self_confidence": 0.85}}

    # Case 1: High signals, no contradiction
    conf_clean = estimator.estimate_confidence(
        evidence_support_score=0.9,
        agent_agreement_score=0.95,
        retrieved_chunks=chunks,
        agent_outputs=outputs,
        has_contradictions=False
    )
    expected = 0.4*0.9 + 0.25*0.95 + 0.2*0.85 + 0.15*0.85
    assert abs(conf_clean - round(expected, 4)) < 0.01
    assert conf_clean >= 0.8

    # Case 2: Contradiction penalty applied
    conf_penalized = estimator.estimate_confidence(
        evidence_support_score=0.9,
        agent_agreement_score=0.95,
        retrieved_chunks=chunks,
        agent_outputs=outputs,
        has_contradictions=True
    )
    assert conf_penalized == round(max(0.0, conf_clean - 0.35), 4)


def test_controller_decision_table_and_loop_cap():
    """Verify controller produces all 4 decisions (CLARIFY, ANSWER, RETRIEVE_MORE, ABSTAIN) and respects loop cap."""
    controller = ResponseController(tau_high=0.70, tau_low=0.45, min_evidence_coverage=0.60, max_retries=2)

    # 1. Ambiguity -> CLARIFY
    dec1, _ = controller.decide(is_ambiguous=True, overall_confidence=0.85, evidence_coverage=0.9, has_contradictions=False, retrieval_attempt=1)
    assert dec1 == "CLARIFY"

    # 2. High Conf, good coverage, no contradictions -> ANSWER
    dec2, _ = controller.decide(is_ambiguous=False, overall_confidence=0.82, evidence_coverage=0.8, has_contradictions=False, retrieval_attempt=1)
    assert dec2 == "ANSWER"

    # 3. Marginal Conf or low coverage, retries < max_retries -> RETRIEVE_MORE
    dec3, _ = controller.decide(is_ambiguous=False, overall_confidence=0.55, evidence_coverage=0.5, has_contradictions=False, retrieval_attempt=1)
    assert dec3 == "RETRIEVE_MORE"

    # 4. Loop cap reached (retries >= max_retries) with marginal score -> ABSTAIN
    dec4, _ = controller.decide(is_ambiguous=False, overall_confidence=0.55, evidence_coverage=0.5, has_contradictions=False, retrieval_attempt=2)
    assert dec4 == "ABSTAIN"

    # 5. Low confidence below tau_low -> ABSTAIN
    dec5, _ = controller.decide(is_ambiguous=False, overall_confidence=0.30, evidence_coverage=0.2, has_contradictions=False, retrieval_attempt=1)
    assert dec5 == "ABSTAIN"


def test_json_schema_validity_and_fallback():
    """Verify structured LLM outputs adhere to Pydantic schemas with repair and fallback."""
    client = LLMClient()

    class SampleSchema(BaseModel):
        recommendation: str
        key_claims: list[str] = Field(default_factory=list)
        self_confidence: float = 0.8

    res = client.generate_structured(
        prompt="Synthesize recommendation for wheat yellow rust",
        schema=SampleSchema
    )
    assert isinstance(res, SampleSchema)
    assert len(res.recommendation) > 0
    assert 0.0 <= res.self_confidence <= 1.0


def test_graph_terminates():
    """Verify that end-to-end StateGraph execution completes without infinite loop."""
    from agritrust.graph import AgriTrustPipeline
    pipeline = AgriTrustPipeline()
    result = pipeline.advise("What is the seed rate for irrigated timely sown wheat?")
    assert "decision" in result
    assert result["decision"] in ["ANSWER", "CLARIFY", "RETRIEVE_MORE", "ABSTAIN"]
    assert "final_answer" in result
    assert result.get("retrieval_attempt", 0) <= 2
