from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from .config import TradingConfig
from .data_models import AllocationDecision


class AllocationSchema(BaseModel):
    reasoning: str = Field(description="Short explanation for allocation choices.")
    weights: dict[str, float] = Field(description="Target weights per industry")


class TradingAgent:
    def __init__(self, llm, trading_cfg: TradingConfig):
        self.llm = llm
        self.trading_cfg = trading_cfg
        self.parser = JsonOutputParser(pydantic_object=AllocationSchema)

    def decide(
        self,
        as_of_date: date,
        industries: list[str],
        market_summary: str,
        event_summary: str,
        report_summary: str,
        current_positions: dict[str, float],
        nav_history: list[float],
    ) -> AllocationDecision:
        format_instructions = self.parser.get_format_instructions()

        system_prompt = (
            "You are a disciplined buy-side portfolio manager. "
            "Task: allocate capital across industries to maximize risk-adjusted return over the next period. "
            "Only use the supplied information. Avoid overfitting and excessive turnover."
        )

        user_prompt = f"""
Date: {as_of_date}
Industries: {industries}
Risk constraints:
- max single industry position: {self.trading_cfg.max_position:.2f}
- long-only (weights >= 0)
- total weight should be <= 1.0

Current positions:
{json.dumps(current_positions, ensure_ascii=False)}

Recent NAV history (tail):
{nav_history[-20:]}

Market summary:
{market_summary}

Major events summary:
{event_summary}

Broker report summary:
{report_summary}

Return JSON only.
{format_instructions}
""".strip()

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        raw = self.llm.invoke(messages)

        parsed = self.parser.parse(raw.content)
        weights = self._post_process_weights(parsed.get("weights", {}), industries)
        reasoning = str(parsed.get("reasoning", ""))
        return AllocationDecision(date=as_of_date, weights=weights, reasoning=reasoning)

    def _post_process_weights(self, weights: dict[str, float], industries: list[str]) -> dict[str, float]:
        out = {industry: max(0.0, float(weights.get(industry, 0.0))) for industry in industries}
        total = sum(out.values())
        if total > 1e-12:
            out = {k: v / total for k, v in out.items()}
        for k, v in list(out.items()):
            out[k] = min(v, self.trading_cfg.max_position)
        capped_total = sum(out.values())
        if capped_total > 1.0:
            out = {k: v / capped_total for k, v in out.items()}
        return out
