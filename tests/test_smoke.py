"""Smoke test running end-to-end AgriTrust pipeline on sample agricultural queries."""

import os
import pytest

os.environ["LLM_PROVIDER"] = "mock"

from agritrust.graph import AgriTrustPipeline


def test_smoke_sample_queries():
    """Verify that AgriTrust executes end-to-end on 3 distinct queries returning valid decisions."""
    pipeline = AgriTrustPipeline()

    test_queries = [
        # 1. Answerable wheat inquiry
        "What is the recommended seed rate and fertilizer for timely sown wheat in Punjab?",
        # 2. Ambiguous query with missing critical information
        "How much chemical pesticide should I spray on my field today?",
        # 3. Unanswerable query outside knowledge corpus
        "What is the pruning schedule for commercial avocado orchards in Tasmania?"
    ]

    valid_decisions = {"ANSWER", "CLARIFY", "RETRIEVE_MORE", "ABSTAIN"}

    for query in test_queries:
        result = pipeline.advise(query)
        assert "decision" in result
        assert result["decision"] in valid_decisions
        assert "overall_confidence" in result
        assert 0.0 <= result["overall_confidence"] <= 1.0
        assert "final_answer" in result
        assert len(result["final_answer"]) > 10
