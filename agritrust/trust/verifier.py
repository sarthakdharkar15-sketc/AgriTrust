"""Evidence Support Verifier (Trust Layer A).
Computes entailment and contradiction scores for candidate claims against retrieved chunks.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("agritrust.trust.verifier")


class EvidenceVerifier:
    """Evaluates NLI entailment of generated claims against retrieved evidence chunks."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        cache_dir: str = ".cache/nli",
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.55,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load CrossEncoder if available, otherwise prepare fast deterministic fallback."""
        import os
        use_offline = os.environ.get("OFFLINE_MODE") == "1" or os.environ.get("LLM_PROVIDER") == "mock"
        if not use_offline:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder NLI model {self.model_name}...")
                self.model = CrossEncoder(self.model_name, max_length=512)
            except Exception as e:
                logger.info(f"CrossEncoder offline or unavailable ({e}). Using deterministic lexical-semantic NLI evaluator.")
                self.model = None
        else:
            self.model = None

    def _predict_nli_pair(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """Predict (contradiction, neutral, entailment) probabilities for a (premise, hypothesis) pair."""
        if self.model is not None:
            try:
                # CrossEncoder typically outputs 3 logits: [contradiction, neutral, entailment] or similar
                logits = self.model.predict([(premise, hypothesis)])[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                # nli-deberta-v3-small labels: 0: contradiction, 1: neutral, 2: entailment
                if len(probs) == 3:
                    return float(probs[0]), float(probs[1]), float(probs[2])
            except Exception as e:
                logger.warning(f"CrossEncoder inference failed: {e}. Using fallback.")

        # Deterministic semantic-lexical fallback
        prem_lower = premise.lower()
        hypo_lower = hypothesis.lower()

        # Check explicit negation / contradiction signals
        negations = ["do not", "never", "avoid", "harmful", "toxic", "prohibited", "suppress"]
        has_prem_neg = any(neg in prem_lower for neg in negations)
        has_hypo_neg = any(neg in hypo_lower for neg in negations)

        prem_words = set(w.strip(".,;:?!()") for w in prem_lower.split() if len(w) > 3)
        hypo_words = set(w.strip(".,;:?!()") for w in hypo_lower.split() if len(w) > 3)

        if not hypo_words:
            return 0.1, 0.8, 0.1

        overlap = len(prem_words.intersection(hypo_words)) / len(hypo_words)

        if has_prem_neg != has_hypo_neg and overlap > 0.4:
            # Likely contradiction
            p_contra = min(0.92, 0.5 + overlap * 0.4)
            p_entail = 0.05
            p_neutral = 1.0 - (p_contra + p_entail)
            return p_contra, p_neutral, p_entail

        # Entailment probability proportional to semantic overlap
        p_entail = min(0.96, overlap * 1.1)
        p_contra = 0.05 if overlap > 0.3 else 0.1
        p_neutral = max(0.0, 1.0 - (p_entail + p_contra))
        return p_contra, p_neutral, p_entail

    def verify_claims(
        self,
        claims: List[str],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Verify each claim against retrieved evidence chunks.
        Returns:
            - claim_verifications: list of per-claim verification results
            - ESS: Evidence Support Score (mean support score across claims)
            - coverage: fraction of claims supported
        """
        if not claims:
            return [], 0.0, 0.0

        if not retrieved_chunks:
            # All unsupported if no evidence chunks available
            verifications = [
                {
                    "claim": claim,
                    "label": "unsupported",
                    "support_score": 0.0,
                    "contradiction_score": 0.0,
                    "best_chunk_id": None,
                    "best_chunk_text": None,
                }
                for claim in claims
            ]
            return verifications, 0.0, 0.0

        verifications = []
        supported_count = 0
        total_support = 0.0

        for claim in claims:
            best_entailment = 0.0
            worst_contradiction = 0.0
            best_chunk_id = None
            best_chunk_text = None

            for chk in retrieved_chunks:
                premise = chk.get("text", "")
                p_contra, p_neutral, p_entail = self._predict_nli_pair(premise, claim)

                if p_entail > best_entailment:
                    best_entailment = p_entail
                    best_chunk_id = chk.get("chunk_id")
                    best_chunk_text = premise[:200] + "..."

                if p_contra > worst_contradiction:
                    worst_contradiction = p_contra

            # Determine label
            if worst_contradiction >= self.contradiction_threshold and worst_contradiction > best_entailment:
                label = "contradicted"
            elif best_entailment >= self.entailment_threshold:
                label = "supported"
                supported_count += 1
            else:
                label = "unsupported"

            total_support += best_entailment

            verifications.append({
                "claim": claim,
                "label": label,
                "support_score": round(best_entailment, 4),
                "contradiction_score": round(worst_contradiction, 4),
                "best_chunk_id": best_chunk_id,
                "best_chunk_text": best_chunk_text,
            })

        num_claims = len(claims)
        ess = round(total_support / num_claims, 4) if num_claims > 0 else 0.0
        coverage = round(supported_count / num_claims, 4) if num_claims > 0 else 0.0

        return verifications, ess, coverage
