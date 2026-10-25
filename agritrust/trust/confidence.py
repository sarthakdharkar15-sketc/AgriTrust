"""Confidence Estimator for AgriTrust multi-agent framework.
Computes calibrated overall confidence score incorporating evidence support, inter-agent agreement,
retrieval relevance, and self-confidence, while penalizing contradictions.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ConfidenceEstimator:
    """Computes composite calibrated confidence score from evidence and agreement signals."""

    def __init__(
        self,
        w1_ess: float = 0.40,
        w2_agr: float = 0.25,
        w3_retrieval: float = 0.20,
        w4_self_conf: float = 0.15,
        contradiction_penalty: float = 0.35,
    ):
        self.w1 = w1_ess
        self.w2 = w2_agr
        self.w3 = w3_retrieval
        self.w4 = w4_self_conf
        self.contradiction_penalty = contradiction_penalty

    def estimate_confidence(
        self,
        evidence_support_score: float,
        agent_agreement_score: float,
        retrieved_chunks: List[Dict[str, Any]],
        agent_outputs: Dict[str, Dict[str, Any]],
        has_contradictions: bool = False,
    ) -> float:
        """Calculate overall confidence score in [0.0, 1.0]."""
        # Mean retrieval score
        if retrieved_chunks:
            retrieval_scores = [chk.get("score", 0.5) for chk in retrieved_chunks]
            mean_retrieval = float(np.mean(retrieval_scores))
        else:
            mean_retrieval = 0.0

        # Mean agent self-confidence
        if agent_outputs:
            self_conf_scores = [out.get("self_confidence", 0.8) for out in agent_outputs.values()]
            mean_self_conf = float(np.mean(self_conf_scores))
        else:
            mean_self_conf = 0.5

        # Raw weighted composite confidence
        raw_conf = (
            self.w1 * evidence_support_score
            + self.w2 * agent_agreement_score
            + self.w3 * mean_retrieval
            + self.w4 * mean_self_conf
        )

        # Apply contradiction penalty
        penalty = self.contradiction_penalty if has_contradictions else 0.0
        final_conf = max(0.0, min(1.0, round(raw_conf - penalty, 4)))

        return final_conf
