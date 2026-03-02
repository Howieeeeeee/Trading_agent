from __future__ import annotations

import json
import re
from collections import defaultdict

from langchain_core.messages import AIMessage

POS_KEYWORDS = ["上修", "增长", "改善", "突破", "利好", "修复", "景气"]
NEG_KEYWORDS = ["下修", "下滑", "承压", "利空", "回落", "风险", "波动"]


class RuleBasedChatModel:
    """A minimal offline model adapter with an invoke(messages) interface."""

    def invoke(self, messages):
        user_text = messages[-1].content if messages else ""
        industries = _parse_list_field(user_text, "Industries")
        market_scores = _parse_market_scores(user_text, industries)
        event_scores = _parse_event_scores(user_text, industries)

        combined = {i: max(0.0, market_scores[i] + event_scores[i]) for i in industries}
        total = sum(combined.values())
        if total <= 1e-12:
            w = 1.0 / max(len(industries), 1)
            weights = {i: w for i in industries}
        else:
            weights = {i: combined[i] / total for i in industries}

        reasoning = "rule_based: weighted by momentum and event sentiment"
        return AIMessage(
            content=json.dumps(
                {
                    "reasoning": reasoning,
                    "weights": weights,
                },
                ensure_ascii=False,
            )
        )


def _parse_list_field(text: str, key: str) -> list[str]:
    m = re.search(rf"{key}:\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    parts = [p.strip().strip("'\"") for p in body.split(",")]
    return [p for p in parts if p]


def _parse_market_scores(text: str, industries: list[str]) -> dict[str, float]:
    scores = {i: 0.01 for i in industries}
    for line in text.splitlines():
        if not line.startswith("-"):
            continue
        for i in industries:
            if f"- {i}:" not in line:
                continue
            m = re.search(r"window_return=([+-]?\d+(?:\.\d+)?)%", line)
            if m:
                val = float(m.group(1)) / 100.0
                scores[i] = max(0.0, val + 0.02)
    return scores


def _parse_event_scores(text: str, industries: list[str]) -> dict[str, float]:
    block = _extract_block(text, "Major events summary:", "Broker report summary:")
    score = defaultdict(float)
    current = None
    for raw in block.splitlines():
        line = raw.strip()
        for i in industries:
            if line.startswith(f"- {i}:"):
                current = i
        if current is None:
            continue
        for kw in POS_KEYWORDS:
            if kw in line:
                score[current] += 0.02
        for kw in NEG_KEYWORDS:
            if kw in line:
                score[current] -= 0.02
    return {i: score[i] for i in industries}


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        end = len(text)
    return text[start:end]
