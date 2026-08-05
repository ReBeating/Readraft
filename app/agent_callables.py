"""Provider-callable definitions shared by tools and Agent actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Type

from pydantic import BaseModel


CallableCategory = Literal[
    "workspace_tool",
    "external_tool",
    "agent_action",
]


@dataclass(frozen=True)
class AgentCallableSpec:
    """Describe one callable without pretending every callable is a tool."""

    name: str
    label: str
    description: str
    input_model: Type[BaseModel]
    category: CallableCategory
    read_only: bool = True

    def native_schema(self) -> dict[str, Any]:
        """Translate the callable to the provider's native `tools` protocol."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }
