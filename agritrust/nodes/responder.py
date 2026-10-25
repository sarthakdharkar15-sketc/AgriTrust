"""Responder node: format final output with evidence citations, decisions, and claim filtering."""

from typing import Dict, Any, List
from agritrust.state import AgriTrustState


def responder_node(state: AgriTrustState) -> Dict[str, Any]:
    """Generate final validated advisory response, dropping unsupported claims for ANSWER."""
    decision = state.get("decision", "ABSTAIN")
    reasons = state.get("decision_reasons", [])
    confidence = state.get("overall_confidence", 0.0)
    verifications = state.get("claim_verifications", [])
    clarifying_q = state.get("clarifying_question")
    retrieved_chunks = state.get("retrieved_chunks", [])

    supported_claims = []
    dropped_claims = []
    active_citations = set()

    for v in verifications:
        claim_text = v.get("claim", "")
        label = v.get("label", "unsupported")
        if label == "supported":
            supported_claims.append(claim_text)
            if v.get("best_chunk_id"):
                active_citations.add(v.get("best_chunk_id"))
        else:
            dropped_claims.append(f"{claim_text} ({label})")

    # Build response per decision
    if decision == "CLARIFY":
        prompt_text = clarifying_q or "To provide an accurate advisory, please specify your crop, growth stage, or local symptoms."
        final_answer = (
            f"**Information Needed Before Advising:**\n\n{prompt_text}\n\n"
            f"> *AgriTrust Clarification Trigger*: The query lacks critical contextual parameters required for safe agronomic recommendations."
        )

    elif decision == "ANSWER":
        if not supported_claims:
            # Safety fallback: if no claims survived verification, abstain rather than hallucinate
            decision = "ABSTAIN"
            reasons.append("All candidate claims were unsupported by retrieved evidence chunks.")
            final_answer = (
                "**Notice**: Although candidate advice was generated, none of the specific claims could be "
                "rigorously verified against authoritative agronomic evidence. AgriTrust has abstained to protect against inaccurate recommendations."
            )
        else:
            claims_formatted = "\n".join([f"- {claim}" for claim in supported_claims])
            citations_str = ", ".join([f"`[{cid}]`" for cid in sorted(list(active_citations))]) if active_citations else "Authoritative Extension Corpus"

            final_answer = (
                f"### AgriTrust Recommendation (Verified)\n\n"
                f"{claims_formatted}\n\n"
                f"---\n"
                f"**Evidence Citations**: {citations_str}\n"
                f"**Overall Confidence**: {confidence * 100:.1f}%\n"
            )

    elif decision == "ABSTAIN":
        reasons_bullet = "\n".join([f"- {r}" for r in reasons])
        final_answer = (
            f"### AgriTrust Abstention Notice\n\n"
            f"AgriTrust cannot reliably provide a verified recommendation for this query with high confidence.\n\n"
            f"**Evaluation Diagnostics:**\n"
            f"{reasons_bullet}\n\n"
            f"**Confidence**: {confidence * 100:.1f}%\n\n"
            f"> *Recommendation*: Consult a certified local agricultural extension officer or provide additional crop and soil parameters."
        )

    else:  # RETRIEVE_MORE or unexpected
        final_answer = (
            f"Retrieval expansion required (Confidence: {confidence * 100:.1f}%). Searching additional corpus segments..."
        )

    return {
        "decision": decision,
        "decision_reasons": reasons,
        "final_answer": final_answer,
        "supported_claims": supported_claims,
        "dropped_claims": dropped_claims,
    }
