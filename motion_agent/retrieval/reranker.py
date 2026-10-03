"""Rule-based structured reranking for retrieval."""

from __future__ import annotations

from motion_agent.retrieval.schemas import RankedReference, RetrievalCandidate, RetrievalQuery


class StructuredReranker:
    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        *,
        query: RetrievalQuery,
        target_duration_s: float | None,
        top_k: int,
    ) -> list[RankedReference]:
        ranked: list[RankedReference] = []
        for candidate in candidates:
            components = {
                "tmr": candidate.tmr_score,
                "action": _contains_score(candidate.caption, query.action),
                "body": _list_contains_score(candidate.caption, query.body_parts),
                "style": _list_contains_score(candidate.caption, query.style),
                "direction": _contains_score(candidate.caption, query.direction),
                "duration": _duration_score(candidate.duration_s, target_duration_s),
                "mismatch_penalty": _mismatch_penalty(candidate.caption, query),
            }
            score = (
                0.65 * components["tmr"]
                + 0.12 * components["action"]
                + 0.06 * components["body"]
                + 0.08 * components["style"]
                + 0.05 * components["direction"]
                + 0.04 * components["duration"]
                - 0.25 * components["mismatch_penalty"]
            )
            ranked.append(
                RankedReference(candidate=candidate, final_score=score, component_scores=components, rank=0)
            )
        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(ranked[:top_k])]


def _contains_score(text: str, value: str | None) -> float:
    if not value:
        return 0.0
    normalized = text.lower().replace("_", " ")
    alternatives = {value.lower().replace("_", " ")}
    if value == "walk":
        alternatives.update({"walks", "walking", "gait"})
    if value == "sit_down":
        alternatives.update({"sit", "sits", "sitting", "seated"})
    if value == "wave":
        alternatives.update({"wave", "waves", "waving"})
    if value == "limping":
        alternatives.update({"limp", "limps", "uneven", "injured"})
    return 1.0 if any(alt in normalized for alt in alternatives) else 0.0


def _list_contains_score(text: str, values: list[str]) -> float:
    if not values:
        return 0.0
    scores = [_contains_score(text, value) for value in values]
    return sum(scores) / len(scores)


def _duration_score(candidate_duration: float | None, target_duration: float | None) -> float:
    if candidate_duration is None or target_duration is None:
        return 0.0
    ratio = min(candidate_duration, target_duration) / max(candidate_duration, target_duration, 1e-6)
    return max(0.0, min(1.0, ratio))


def _mismatch_penalty(text: str, query: RetrievalQuery) -> float:
    lowered = text.lower()
    penalty = 0.0
    opposites = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
    }
    if query.direction and opposites.get(query.direction) in lowered:
        penalty += 1.0
    if query.action == "sit_down" and any(value in lowered for value in ["jump", "walking", "walks"]):
        penalty += 0.5
    if query.action == "jump" and any(value in lowered for value in ["sit", "seated"]):
        penalty += 0.5
    return penalty
