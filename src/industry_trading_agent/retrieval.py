from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from .data_models import EventRecord, ReportDoc


class ContextRetriever:
    def __init__(
        self,
        reports: list[ReportDoc],
        events: list[EventRecord],
        market_data: dict[str, pd.DataFrame],
        market_index: pd.DataFrame | None = None,
    ):
        self.reports = reports
        self.events = events
        self.market_data = market_data
        self.market_index = market_index

    def build_context(
        self,
        as_of_date: date,
        industries: list[str],
        lookback_days: int,
        max_reports_per_industry: int = 3,
        max_events_per_industry: int = 5,
    ) -> dict:
        market_summary = self._market_summary(as_of_date, industries, lookback_days)
        event_summary, event_refs = self._event_summary(as_of_date, industries, lookback_days, max_events_per_industry)
        report_summary, report_refs = self._report_summary(as_of_date, industries, lookback_days, max_reports_per_industry)
        return {
            "market_summary": market_summary,
            "event_summary": event_summary,
            "report_summary": report_summary,
            "event_refs": event_refs,
            "report_refs": report_refs,
        }

    def _market_summary(self, as_of_date: date, industries: list[str], lookback_days: int) -> str:
        lines: list[str] = []
        start_date = as_of_date - timedelta(days=lookback_days)
        dt = pd.Timestamp(as_of_date)

        for industry in industries:
            df = self.market_data[industry]
            window = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= dt)].copy()
            if window.empty:
                lines.append(f"- {industry}: no market data in window")
                continue

            close = window["close"]
            ret = (close.iloc[-1] / close.iloc[0] - 1.0) if len(close) > 1 else 0.0
            vol = window["ret_1d"].std() * (252 ** 0.5)
            lines.append(
                f"- {industry}: window_return={ret:.2%}, annualized_vol~{vol:.2%}, "
                f"last_close={close.iloc[-1]:.4f}, points={len(window)}"
            )

        if self.market_index is not None:
            idx_window = self.market_index[
                (self.market_index["date"] >= pd.Timestamp(start_date)) & (self.market_index["date"] <= dt)
            ].copy()
            if not idx_window.empty:
                idx_ret = idx_window["ret_1d"].sum()
                lines.append(f"- benchmark(HS300): window_return~{idx_ret:.2%}, points={len(idx_window)}")
        return "\n".join(lines)

    def _event_summary(
        self, as_of_date: date, industries: list[str], lookback_days: int, max_events_per_industry: int
    ) -> tuple[str, list[dict]]:
        start_date = as_of_date - timedelta(days=lookback_days)
        grouped: dict[str, list[EventRecord]] = defaultdict(list)
        industries_set = set(industries)
        for event in self.events:
            if event.industry not in industries_set:
                continue
            if start_date <= event.event_date <= as_of_date:
                grouped[event.industry].append(event)

        lines: list[str] = []
        refs: list[dict] = []
        for industry in industries:
            items = sorted(grouped[industry], key=lambda e: e.event_date, reverse=True)[:max_events_per_industry]
            if not items:
                lines.append(f"- {industry}: no major events")
                continue
            lines.append(f"- {industry}:")
            for e in items:
                content = _clean_snippet(e.content)
                lines.append(f"  [{e.event_date}] {e.title} | {content}")
                refs.append(
                    {
                        "industry": industry,
                        "event_date": e.event_date.isoformat(),
                        "title": e.title,
                        "report_date": e.report_date.isoformat() if e.report_date else None,
                        "confidence": e.confidence,
                    }
                )
        return "\n".join(lines), refs

    def _report_summary(
        self, as_of_date: date, industries: list[str], lookback_days: int, max_reports_per_industry: int
    ) -> tuple[str, list[dict]]:
        start_date = as_of_date - timedelta(days=lookback_days)
        grouped: dict[str, list[ReportDoc]] = defaultdict(list)
        industries_set = set(industries)
        for report in self.reports:
            if report.industry not in industries_set:
                continue
            if start_date <= report.report_date <= as_of_date:
                grouped[report.industry].append(report)

        lines: list[str] = []
        refs: list[dict] = []
        for industry in industries:
            items = sorted(grouped[industry], key=lambda r: r.report_date, reverse=True)[:max_reports_per_industry]
            if not items:
                lines.append(f"- {industry}: no recent reports")
                continue
            lines.append(f"- {industry}:")
            for r in items:
                content = _clean_snippet(r.content)
                lines.append(f"  [{r.report_date}] report_id={r.report_id} | {content}")
                refs.append(
                    {
                        "industry": industry,
                        "report_id": r.report_id,
                        "report_date": r.report_date.isoformat(),
                        "file_path": r.file_path,
                    }
                )
        return "\n".join(lines), refs


def _clean_snippet(text: str, max_len: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."
