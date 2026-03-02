from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class ReportDoc:
    report_id: str
    industry: str
    report_date: date
    file_path: str
    content: str


@dataclass
class EventRecord:
    event_date: date
    industry: str
    title: str
    content: str
    report_date: date | None = None
    confidence: float | None = None


@dataclass
class DecisionContext:
    as_of_date: date
    industries: list[str]
    market_summary: str
    event_summary: str
    report_summary: str
    current_positions: dict[str, float]
    nav_history: list[float]


@dataclass
class AllocationDecision:
    date: date
    weights: dict[str, float]
    reasoning: str
