"""Normalize bounded Pydantic schemas for strict provider output."""

from copy import deepcopy

from motion_agent.llm.errors import LLMSchemaError


def strict_output_schema(model) -> dict:
    schema = deepcopy(model.model_json_schema())

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                if "properties" not in node:
                    raise LLMSchemaError("LLM_UNBOUNDED_OUTPUT_OBJECT")
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for child in node.values():
                visit(child)

    visit(schema)
    return schema
