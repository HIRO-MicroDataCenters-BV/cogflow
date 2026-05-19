"""Condition Agent node — LLM-driven router."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from ..ir.model import IRNode
from .base import NodeFactory


class ConditionAgentFactory(NodeFactory):
    ir_type = "condition_agent"

    def __init__(
        self,
        *,
        model: Any = None,
        instructions: str | None = None,
        scenarios: list[dict[str, str]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            conditionAgentModel=model if isinstance(model, str) else "chatOpenAI",
            conditionAgentInstructions=instructions or "",
            conditionAgentScenarios=scenarios or [],
            **extra,
        )
        self._model = model
        self._scenarios = scenarios or []

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        raw_model = getattr(self, "_model", None)
        model = raw_model if not isinstance(raw_model, str) else None
        scenarios = list(getattr(self, "_scenarios", None) or self.config.get("conditionAgentScenarios") or [])
        instructions = self.config.get("conditionAgentInstructions", "") or ""

        def condition_agent_node(state: dict[str, Any]) -> dict[str, Any]:
            if model is None or not scenarios:
                return {"_condition_branch": "default"}
            scenario_text = "\n".join(
                f"- {s.get('name', s.get('output', 'scenario'))}: {s.get('description', '')}"
                for s in scenarios
            )
            prompt = [
                SystemMessage(content=f"{instructions}\nChoose exactly one scenario name from:\n{scenario_text}"),
                *(state.get("messages") or []),
                HumanMessage(content="Reply with only the scenario name."),
            ]
            response = model.invoke(prompt)
            choice = (getattr(response, "content", "") or "").strip().casefold()
            names = [s.get("name", s.get("output")) for s in scenarios if s.get("name", s.get("output"))]
            # Prefer an exact match (case-insensitive) on the model's reply so
            # overlapping scenario names like "approve" / "approved" don't
            # silently route to the shorter prefix.
            for name in names:
                if name.casefold() == choice:
                    return {"_condition_branch": name}
            # Fall back to whole-word containment, then to the first scenario.
            for name in names:
                if name.casefold() in choice.split():
                    return {"_condition_branch": name}
            return {"_condition_branch": names[0] if names else "default"}

        return condition_agent_node


def condition_agent(**kwargs: Any) -> ConditionAgentFactory:
    return ConditionAgentFactory(**kwargs)
