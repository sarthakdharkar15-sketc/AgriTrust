"""Agent Agreement Analyzer (Trust Layer B).
Evaluates pairwise agreement and flags conflicts across specialized agent claims.
"""

from typing import Dict, List, Any, Tuple
import logging
import itertools
import numpy as np

logger = logging.getLogger("agritrust.trust.agreement")


class AgreementAnalyzer:
    """Computes pairwise semantic alignment and detects contradictions between agent outputs."""

    def __init__(self, contradiction_threshold: float = 0.55):
        self.contradiction_threshold = contradiction_threshold

    def _calc_similarity_and_nli(self, claim_a: str, claim_b: str) -> Tuple[float, float]:
        """Compute semantic overlap similarity and contradiction probability between two claims."""
        words_a = set(w.strip(".,;:?!()'\"").lower() for w in claim_a.split() if len(w) > 3)
        words_b = set(w.strip(".,;:?!()'\"").lower() for w in claim_b.split() if len(w) > 3)

        min_len = min(len(words_a), len(words_b))
        overlap = len(words_a.intersection(words_b)) / min_len if min_len > 0 else 0.0
        sim = min(1.0, 0.45 + 0.55 * overlap)

        # Detect contradictory directives
        negations = ["not", "never", "avoid", "harmful", "toxic", "prohibited", "suppress", "reduce"]
        neg_a = any(n in claim_a.lower() for n in negations)
        neg_b = any(n in claim_b.lower() for n in negations)

        contradiction = 0.0
        if neg_a != neg_b and overlap > 0.25:
            contradiction = min(0.95, 0.60 + overlap * 0.35)

        return sim, contradiction

    def analyze_agreement(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
        """Compute inter-agent agreement score (AGR in [0, 1]) and list detected conflicts."""
        active_agents = list(agent_outputs.keys())
        if len(active_agents) < 2:
            # Single agent: baseline agreement is high, no conflicts
            return 1.0, []

        pairwise_agreements = []
        conflicts = []

        pairs = list(itertools.combinations(active_agents, 2))
        for agent_a, agent_b in pairs:
            claims_a = agent_outputs[agent_a].get("key_claims", [])
            claims_b = agent_outputs[agent_b].get("key_claims", [])

            if not claims_a or not claims_b:
                pairwise_agreements.append(0.7)
                continue

            # For each claim in A, find best alignment in B; and vice-versa
            best_sims_a = []
            for ca in claims_a:
                sims_for_ca = []
                for cb in claims_b:
                    sim, contra = self._calc_similarity_and_nli(ca, cb)
                    sims_for_ca.append(sim)
                    if contra >= self.contradiction_threshold:
                        conflicts.append({
                            "agent_a": agent_a,
                            "agent_b": agent_b,
                            "claim_a": ca,
                            "claim_b": cb,
                            "nli_contradiction_score": round(contra, 4),
                            "description": f"Conflict between {agent_a} and {agent_b} on contradictory directives.",
                        })
                best_sims_a.append(max(sims_for_ca) if sims_for_ca else 0.5)

            best_sims_b = []
            for cb in claims_b:
                sims_for_cb = [self._calc_similarity_and_nli(ca, cb)[0] for ca in claims_a]
                best_sims_b.append(max(sims_for_cb) if sims_for_cb else 0.5)

            pair_score = float(np.mean(best_sims_a + best_sims_b))
            pairwise_agreements.append(pair_score)

        mean_agreement = float(np.mean(pairwise_agreements)) if pairwise_agreements else 1.0

        # Penalize agreement score if direct conflicts exist
        conflict_penalty = min(0.5, len(conflicts) * 0.25)
        agr = max(0.0, min(1.0, round(mean_agreement - conflict_penalty, 4)))

        return agr, conflicts
