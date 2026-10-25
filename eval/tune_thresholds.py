"""Dev-set threshold and weight grid search for AgriTrust response controller."""

import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, Any, List
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agritrust.eval.tune")


def tune_on_dev(dev_file: str = "eval/dev.jsonl", config_path: str = "agritrust/config.yaml"):
    """Perform grid search on dev set to optimize tau_high, tau_low, and min_evidence_coverage."""
    dev_path = Path(dev_file)
    if not dev_path.exists():
        from eval.generate_dataset import validate_and_save_dataset
        validate_and_save_dataset()

    items = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    from agritrust.graph import AgriTrustPipeline

    pipeline = AgriTrustPipeline(config_path)

    logger.info(f"Precomputing dev set outputs for {len(items)} items...")
    cached_signals = []
    for item in items:
        res = pipeline.advise(item["question"])
        cached_signals.append({
            "is_ambiguous": res.get("is_ambiguous", False),
            "overall_confidence": res.get("overall_confidence", 0.0),
            "evidence_coverage": res.get("evidence_coverage", 0.0),
            "has_contradictions": any(v.get("label") == "contradicted" for v in res.get("claim_verifications", [])) or len(res.get("agent_conflicts", [])) > 0,
            "retrieval_attempt": res.get("retrieval_attempt", 1),
            "category": item["category"]
        })

    tau_high_grid = [0.65, 0.70, 0.75]
    tau_low_grid = [0.40, 0.45, 0.50]
    coverage_grid = [0.50, 0.60, 0.70]

    best_score = -1.0
    best_params = {}

    for tau_h in tau_high_grid:
        for tau_l in tau_low_grid:
            if tau_l >= tau_h:
                continue
            for cov in coverage_grid:
                pipeline.controller.tau_high = tau_h
                pipeline.controller.tau_low = tau_l
                pipeline.controller.min_evidence_coverage = cov

                correct = 0
                for sig in cached_signals:
                    dec, _ = pipeline.controller.decide(
                        is_ambiguous=sig["is_ambiguous"],
                        overall_confidence=sig["overall_confidence"],
                        evidence_coverage=sig["evidence_coverage"],
                        has_contradictions=sig["has_contradictions"],
                        retrieval_attempt=sig["retrieval_attempt"]
                    )
                    cat = sig["category"]
                    if cat == "answerable" and dec == "ANSWER":
                        correct += 1
                    elif cat == "unanswerable" and dec == "ABSTAIN":
                        correct += 1
                    elif cat == "ambiguous" and dec == "CLARIFY":
                        correct += 1
                    elif cat == "conflicting/misleading premise" and dec in ["ABSTAIN", "CLARIFY"]:
                        correct += 1

                score = correct / len(cached_signals)
                if score > best_score:
                    best_score = score
                    best_params = {"tau_high": tau_h, "tau_low": tau_l, "min_evidence_coverage": cov, "dev_accuracy": score}

    logger.info(f"Optimal Dev Configuration: {best_params}")

    # Write report
    report_path = Path("results/dev_tuning_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"Saved dev tuning report to {report_path}")
    return best_params


if __name__ == "__main__":
    tune_on_dev()
