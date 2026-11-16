from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


def evaluate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in results if r.get("gt_referents_mapped")]
    exact_hits = 0
    any_hits = 0
    true_positive = false_positive = false_negative = 0
    for r in rows:
        gt = set(r.get("gt_referents_mapped") or [])
        pred = set(r.get("selected_object_names") or ([r.get("selected_object_name")] if r.get("selected_object_name") else []))
        true_positive += len(gt & pred)
        false_positive += len(pred - gt)
        false_negative += len(gt - pred)
        if gt & pred: any_hits += 1
        if gt == pred: exact_hits += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(results), "evaluable": len(rows), "coverage": len(rows) / len(results) if results else 0.0,
        "any_hit_count": any_hits, "any_hit_accuracy": any_hits / len(rows) if rows else None,
        "exact_set_match_count": exact_hits, "exact_set_match": exact_hits / len(rows) if rows else None,
        "set_true_positive": true_positive, "set_false_positive": false_positive, "set_false_negative": false_negative,
        "micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
    }


def evaluate_by_scene(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for scene in sorted({str(r.get("scene_key", r.get("scene_id", ""))) for r in results}):
        out[scene] = evaluate_results([r for r in results if str(r.get("scene_key", r.get("scene_id", ""))) == scene])
    return out
