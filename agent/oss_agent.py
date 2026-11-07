"""Target D (optional): an external agent framework, used as published.

smolagents (HuggingFace) drives its own ReAct loop over our tools. This is the
second "someone else's system": we did not design its control flow, so its
failure modes are genuinely external.

If smolagents is not installed the module raises ImportError and every test
that depends on it is skipped — never silently treated as passing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from evals import llm as llm_mod

TOOL_SPEC_NOTE = """You have access to these tools:
- run_smoke(target: str)
- run_regression(target: str)
- run_integration(left: str, right: str)
- run_data_check(target: str)
- ask_clarification(question: str)

Never call a heavy suite (regression/integration) without a clear target.
If the request is ambiguous, call ask_clarification.
"""


def _import_smol():
    try:
        from smolagents import OpenAIServerModel, ToolCallingAgent, tool  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("smolagents is not installed (optional dependency)") from exc
    return True


class OssAgent:
    name = "oss-agent(smolagents)"

    def __init__(self, provider: llm_mod.BaseProvider | None = None, max_steps: int = 4):
        _import_smol()
        from smolagents import OpenAIServerModel, ToolCallingAgent, tool

        self.provider = provider or llm_mod.get_provider()
        if not getattr(self.provider, "is_real", False):
            raise RuntimeError("smolagents needs a real OpenAI-compatible endpoint (ARK)")

        model = OpenAIServerModel(
            model_id=os.environ.get("ARK_CHAT_MODEL", llm_mod.DEFAULT_CHAT),
            api_base=f"{self.provider.base_url}/",
            api_key=self.provider.api_key,
        )

        calls: list[dict[str, Any]] = []
        self.calls = calls

        @tool
        def run_smoke(target: str) -> str:
            """Run the smoke suite for a system."""
            calls.append({"tool": "run_smoke", "args": {"target": target}})
            return f"smoke suite executed for {target}"

        @tool
        def run_regression(target: str) -> str:
            """Run the full regression suite for a system."""
            calls.append({"tool": "run_regression", "args": {"target": target}})
            return f"regression suite executed for {target}"

        @tool
        def run_integration(left: str, right: str) -> str:
            """Run integration verification between two systems."""
            calls.append({"tool": "run_integration", "args": {"left": left, "right": right}})
            return f"integration executed for {left}<->{right}"

        @tool
        def run_data_check(target: str) -> str:
            """Run read-only DB/file reconciliation for a system."""
            calls.append({"tool": "run_data_check", "args": {"target": target}})
            return f"data check executed for {target}"

        @tool
        def ask_clarification(question: str) -> str:
            """Ask the user to clarify before doing anything."""
            calls.append({"tool": "ask_clarification", "args": {"question": question}})
            return "clarification requested"

        self.agent = ToolCallingAgent(tools=[run_smoke, run_regression, run_integration,
                                             run_data_check, ask_clarification],
                                      model=model, max_steps=max_steps)

    def route(self, user_input: str) -> dict[str, Any]:
        """Run the framework's own loop, then map observed tool calls back to intents."""
        self.calls.clear()
        try:
            self.agent.run(user_input)
        except Exception as exc:  # noqa: BLE001 - framework internals vary
            return {"intent": "error", "target": "unknown", "reason": str(exc)[:150]}
        return self._to_route()

    def _to_route(self) -> dict[str, Any]:
        tool_to_intent = {
            "run_smoke": "smoke",
            "run_regression": "regression",
            "run_integration": "integration",
            "run_data_check": "data-check",
            "ask_clarification": "clarify",
        }
        if not self.calls:
            return {"intent": "out_of_scope", "target": "unknown", "reason": "no tool was called"}
        first = self.calls[0]
        target = first["args"].get("target") or first["args"].get("left", "unknown")
        return {
            "intent": tool_to_intent.get(first["tool"], "unknown"),
            "target": target,
            "reason": f"first tool: {json.dumps(first['args'], ensure_ascii=False)}",
        }
