"""Small-sample threshold support; deliberately no benchmark claims."""

import math
from statistics import mean


def provisional_calibration(positive_scores: list[float], negative_scores: list[float]) -> dict:
    if not all(math.isfinite(score) for score in positive_scores + negative_scores):
        raise ValueError("calibration scores must be finite")
    separation = bool(positive_scores and negative_scores and min(positive_scores) > max(negative_scores))
    return {"status": "PROVISIONAL_CALIBRATION", "tested_pairs": len(positive_scores) + len(negative_scores),
            "positive_scores": positive_scores, "negative_scores": negative_scores,
            "positive_mean": mean(positive_scores) if positive_scores else None,
            "negative_mean": mean(negative_scores) if negative_scores else None,
            "separation": separation,
            "provisional_threshold": (min(positive_scores) + max(negative_scores)) / 2 if separation else None,
            "production_calibrated": False}
