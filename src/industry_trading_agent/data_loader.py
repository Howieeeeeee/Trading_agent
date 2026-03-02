from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .data_models import EventRecord, ReportDoc

REPORT_NAME_PATTERN = re.compile(r"^(?P<id>[^_]+)_(?P<industry>[^_]+)_(?P<dt>\d{8})")


@dataclass
class LoadedData:
    reports: list[ReportDoc]
    events: list[EventRecord]
    market_data: dict[str, pd.DataFrame]


class DataLoader:
    def __init__(self, reports_dir: Path, events_file: Path, market_dir: Path):
        self.reports_dir = reports_dir
        self.events_file = events_file
        self.market_dir = market_dir

    def load_all(self, industries: list[str]) -> LoadedData:
        reports = self.load_reports(industries)
        events = self.load_events(industries)
        market_data = self.load_market_data(industries)
        return LoadedData(reports=reports, events=events, market_data=market_data)

    def load_reports(self, industries: list[str]) -> list[ReportDoc]:
        industries_set = set(industries)
        docs: list[ReportDoc] = []
        if not self.reports_dir.exists():
            return docs

        candidates = sorted(self.reports_dir.glob("*"))
        for file in candidates:
            if file.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            match = REPORT_NAME_PATTERN.match(file.stem)
            if not match:
                continue
            industry = match.group("industry")
            if industry not in industries_set:
                continue
            report_date = pd.to_datetime(match.group("dt"), format="%Y%m%d").date()
            if file.suffix.lower() == ".pdf":
                content = _read_pdf_text(file)
            else:
                content = file.read_text(encoding="utf-8")
            docs.append(
                ReportDoc(
                    report_id=match.group("id"),
                    industry=industry,
                    report_date=report_date,
                    file_path=str(file),
                    content=content,
                )
            )
        return docs

    def load_events(self, industries: list[str]) -> list[EventRecord]:
        industries_set = set(industries)
        if not self.events_file.exists():
            return []

        suffix = self.events_file.suffix.lower()
        if suffix == ".csv":
            rows = _load_csv(self.events_file)
        elif suffix in {".json", ".jsonl"}:
            rows = _load_json_family(self.events_file)
        else:
            rows = _load_text_events(self.events_file)

        events: list[EventRecord] = []
        for row in rows:
            industry = str(row.get("industry", "")).strip()
            if industry not in industries_set:
                continue
            dt = pd.to_datetime(str(row.get("date", "")), errors="coerce")
            if pd.isna(dt):
                continue
            events.append(
                EventRecord(
                    event_date=dt.date(),
                    industry=industry,
                    title=str(row.get("title", "")),
                    content=str(row.get("content", "")),
                )
            )
        events.sort(key=lambda e: e.event_date)
        return events

    def load_market_data(self, industries: list[str]) -> dict[str, pd.DataFrame]:
        market_data: dict[str, pd.DataFrame] = {}
        for industry in industries:
            file_path = self.market_dir / f"{industry}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing market file for industry '{industry}': {file_path}")

            df = pd.read_csv(file_path)
            required_cols = {"date", "close"}
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f"Market file {file_path} must contain columns: {sorted(required_cols)}")

            df["date"] = pd.to_datetime(df["date"]) 
            df = df.sort_values("date").reset_index(drop=True)
            df["ret_1d"] = df["close"].pct_change().fillna(0.0)
            market_data[industry] = df
        return market_data


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_json_family(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "events" in data and isinstance(data["events"], list):
        return data["events"]
    return []


def _load_text_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: YYYY-MM-DD|industry|title|content
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        rows.append(
            {
                "date": parts[0],
                "industry": parts[1],
                "title": parts[2],
                "content": parts[3],
            }
        )
    return rows


def filter_reports_by_date(reports: list[ReportDoc], end_date: date, lookback_days: int) -> list[ReportDoc]:
    start_date = end_date - pd.Timedelta(days=lookback_days).to_pytimedelta()
    return [r for r in reports if start_date <= r.report_date <= end_date]


def filter_events_by_date(events: list[EventRecord], end_date: date, lookback_days: int) -> list[EventRecord]:
    start_date = end_date - pd.Timedelta(days=lookback_days).to_pytimedelta()
    return [e for e in events if start_date <= e.event_date <= end_date]
