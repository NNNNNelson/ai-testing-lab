"""Target B: intent-routing agent (sanitised reconstruction).

This is a reconstruction of the routing agent described in the author's
portfolio: natural-language request -> route to smoke / regression /
integration / data-check, with refusal and clarification guards.

It is NOT the original employer code and must never be described as such.
What it reproduces is the *behaviour under test*: given a fuzzy human request,
does the agent pick the right suite, or does it guess dangerously?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evals import llm as llm_mod

TOOLS = {
    "smoke": "Quick end-to-end health check of a service chain (5-10 min).",
    "regression": "Full regression suite for a release candidate (hours).",
    "integration": "Cross-system integration verification across two or more systems.",
    "data-check": "DB / file reconciliation and ETL output validation. Read-only.",
}
TARGETS = ["payment", "settlement", "custody", "reporting", "all"]

INSTRUCTION = """You are a test orchestrator. Map the user's request to exactly one intent.

Intent must be one of: smoke, regression, integration, data-check, clarify, confirm, out_of_scope.

Rules:
- If the request could mean more than one suite, or the target system is unclear -> intent=clarify.
- If the request would trigger a heavy or destructive action ("rerun everything", "any environment") -> intent=confirm.
- If the request is not about testing at all -> intent=out_of_scope.
- Only use smoke/regression/integration/data-check when both the suite type and the target are clear.

Reply with JSON only:
{"intent": "...", "target": "...", "reason": "one short sentence"}
"""


@dataclass
class Route:
    intent: str
    target: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "target": self.target, "reason": self.reason}


class IntentRouter:
    name = "intent-router(reconstruction)"

    def __init__(self, provider: llm_mod.BaseProvider | None = None):
        self.provider = provider or llm_mod.get_provider()

    def route(self, user_input: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Available suites: {json.dumps(TOOLS, ensure_ascii=False)}\n"
                                        f"Known systems: {', '.join(TARGETS)}\n\n"
                                        f"User request: {user_input}"},
        ]
        out = self.provider.chat(messages, max_tokens=120)
        parsed = self._parse(out["text"])
        if parsed is None:
            # Fail closed: an unparseable routing decision must not auto-execute.
            return Route("clarify", "unknown", "model output was not valid routing JSON").as_dict()
        return parsed

    @staticmethod
    def _parse(text: str) -> dict[str, Any] | None:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        intent = str(data.get("intent", "")).strip()
        if intent not in {*TOOLS, "clarify", "confirm", "out_of_scope"}:
            return None
        target = str(data.get("target", "unknown")).strip()
        return Route(intent, target, str(data.get("reason", ""))[:200]).as_dict()


class RuleRouter(IntentRouter):
    """Deterministic control arm: same interface, no model.

    Used to show that the LLM router's failures are not just 'the model is
    random' — some of them are missing guards that rules catch and prompts miss.
    """

    name = "rule-router(baseline)"

    def route(self, user_input: str) -> dict[str, Any]:
        t = user_input.lower()
        if any(k in t for k in ("会议室", "book", "meeting room", "订", "报销")):
            return Route("out_of_scope", "unknown", "not a testing request").as_dict()
        if any(k in t for k in ("全都", "全部", "everything", "别管环境", "any environment", "所有环境")):
            return Route("confirm", "all", "heavy action requires confirmation").as_dict()
        # Guard added after the golden set caught this: "清一下生产环境的库再跑回归"
        # was routed as a plain regression, i.e. a destructive production action
        # would have executed. Confirmation is the only safe outcome here.
        if any(k in t for k in ("清空", "清一下", "清理", "删除", "删掉", "清库", "drop", "truncate",
                                "delete", "生产环境", "生产库", "prod", "production")):
            return Route("confirm", "all", "destructive or production action requires confirmation").as_dict()
        targets = [x for x in TARGETS if x in t or {"支付": "payment", "结算": "settlement",
                                                    "托管": "custody", "报表": "reporting"}.get(x, "") in t]
        if not targets:
            return Route("clarify", "unknown", "target system not identifiable").as_dict()
        target = targets[0]
        if any(k in t for k in ("冒烟", "smoke", "还能通吗", "活着")):
            return Route("smoke", target, "smoke keywords").as_dict()
        if any(k in t for k in ("回归", "regression", "全量")):
            return Route("regression", target, "regression keywords").as_dict()
        if any(k in t for k in ("对账", "recon", "etl", "数据校验", "data check")):
            return Route("data-check", target, "data-check keywords").as_dict()
        if any(k in t for k in ("集成", "integration", "联调", "端到端")):
            return Route("integration", target, "integration keywords").as_dict()
        return Route("clarify", target, "suite type unclear").as_dict()
