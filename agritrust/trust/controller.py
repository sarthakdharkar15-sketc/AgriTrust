"""Response Controller for AgriTrust multi-agent framework.
Directs execution between ANSWER, CLARIFY, RETRIEVE_MORE, and ABSTAIN.
"""

from typing import Dict, Any, List, Tuple


class ResponseController:
    """Controls final pipeline decision based on confidence, ambiguity, coverage, and retries."""

    def __init__(
        self,
        tau_high: float = 0.70,
        tau_low: float = 0.45,
        min_evidence_coverage: float = 0.60,
        max_retries: int = 2,
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.min_evidence_coverage = min_evidence_coverage
        self.max_retries = max_retries

    def decide(
        self,
        is_ambiguous: bool,
        overall_confidence: float,
        evidence_coverage: float,
        has_contradictions: bool,
        retrieval_attempt: int,
    ) -> Tuple[str, List[str]]:
        """Evaluate state signals and return (decision, decision_reasons).
        Decisions:
            - 'CLARIFY'
            - 'ANSWER'
            - 'RETRIEVE_MORE'
            - 'ABSTAIN'
        """
        reasons = []

        # 1. Ambiguity flag check
        if is_ambiguous:
            reasons.append("Query lacks critical agricultural details (e.g. crop type, symptoms, or soil context).")
            return "CLARIFY", reasons

        # 2. High confidence and verified consistency
        if overall_confidence >= self.tau_high and not has_contradictions and evidence_coverage >= self.min_evidence_coverage:
            reasons.append(f"Confidence score ({overall_confidence:.2f}) satisfies high threshold (tau_high={self.tau_high}).")
            reasons.append(f"Evidence coverage ({evidence_coverage*100:.1f}%) meets verification standard without contradictions.")
            return "ANSWER", reasons

        # 3. Retrieve more condition (if retries not exhausted and confidence is not critically low)
        can_retry = retrieval_attempt < self.max_retries
        is_marginal_conf = self.tau_low <= overall_confidence < self.tau_high
        is_low_coverage = evidence_coverage < self.min_evidence_coverage

        if can_retry and overall_confidence >= self.tau_low and (is_marginal_conf or is_low_coverage or has_contradictions):
            if is_marginal_conf:
                reasons.append(f"Confidence ({overall_confidence:.2f}) is in marginal range [{self.tau_low}, {self.tau_high}).")
            if is_low_coverage:
                reasons.append(f"Evidence coverage ({evidence_coverage*100:.1f}%) below minimum required ({self.min_evidence_coverage*100:.1f}%).")
            if has_contradictions:
                reasons.append("Contradictions detected between claims or evidence. Seeking additional grounding.")
            reasons.append(f"Initiating expanded retrieval attempt ({retrieval_attempt + 1}/{self.max_retries}).")
            return "RETRIEVE_MORE", reasons

        # 4. Abstain condition
        if overall_confidence < self.tau_low:
            reasons.append(f"Confidence ({overall_confidence:.2f}) falls below low threshold (tau_low={self.tau_low}).")
        if has_contradictions:
            reasons.append("Unresolvable contradictions detected between advisory agents or retrieved evidence.")
        if retrieval_attempt >= self.max_retries:
            reasons.append(f"Maximum retrieval retries ({self.max_retries}) reached without attaining sufficient evidence support.")

        reasons.append("Abstaining from potentially unsafe or ungrounded agricultural recommendation.")
        return "ABSTAIN", reasons
