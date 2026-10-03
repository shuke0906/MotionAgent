"""Coarse semantics built from DSL, without exact count or timeline claims."""

from motion_agent.compiler.schemas import MotionSpecification


def semantic_text(spec: MotionSpecification) -> str:
    descriptions = []
    for segment in spec.segments:
        words = [segment.action.replace("_", " ")]
        words.extend(action.replace("_", " ") for action in segment.secondary_actions)
        words.extend(part.replace("_", " ") for part in segment.body_parts if part != "full_body")
        if segment.direction and segment.direction != "none":
            words.append(segment.direction)
        if segment.speed and segment.speed != "normal":
            words.append(segment.speed.replace("_", " "))
        words.extend(segment.style)
        if segment.orientation and segment.orientation not in words:
            words.append(segment.orientation)
        descriptions.append(" ".join(words))
    return "A person performs these actions: " + "; ".join(descriptions) + "."
