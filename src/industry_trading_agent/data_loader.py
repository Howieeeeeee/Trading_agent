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
BK_FILE_PATTERN = re.compile(r"__(?P<code>BK\d+)\.csv$", re.IGNORECASE)


@dataclass
class LoadedData:
    reports: list[ReportDoc]
    events: list[EventRecord]
    market_data: dict[str, pd.DataFrame]
    market_index: pd.DataFrame | None
    market_files: dict[str, str]


class DataLoader:
    def __init__(
        self,
        report_raw_dir: Path | None,
        report_summary_dir: Path | None,
        events_file: Path,
        market_dir: Path,
        industry_match_file: Path | None = None,
        market_index_file: Path | None = None,
    ):
        self.report_raw_dir = report_raw_dir
        self.report_summary_dir = report_summary_dir
        self.events_file = events_file
        self.market_dir = market_dir
        self.industry_match_file = industry_match_file
        self.market_index_file = market_index_file
        self._last_market_files: dict[str, str] = {}

    def load_all(self, industries: list[str]) -> LoadedData:
        reports = self.load_reports(industries)
        events = self.load_events(industries)
        market_data = self.load_market_data(industries)
        market_index = self.load_market_index()
        return LoadedData(
            reports=reports,
            events=events,
            market_data=market_data,
            market_index=market_index,
            market_files=dict(self._last_market_files),
        )

    def load_reports(self, industries: list[str]) -> list[ReportDoc]:
        industries_set = set(industries)
        docs: list[ReportDoc] = []
        source_dir = self._resolve_report_source_dir()
        if source_dir is None:
            return docs

        candidates = sorted(source_dir.glob("*"))
        for file in candidates:
            if file.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            match = REPORT_NAME_PATTERN.match(file.stem)
            if not match:
                continue
            industry = match.group("industry")
            if industry not in industries_set:
                continue
            report_date = pd.to_datetime(match.group("dt"), format="%Y%m%d", errors="coerce")
            if pd.isna(report_date):
                continue
            content = _read_pdf_text(file) if file.suffix.lower() == ".pdf" else file.read_text(encoding="utf-8")
            docs.append(
                ReportDoc(
                    report_id=match.group("id"),
                    industry=industry,
                    report_date=report_date.date(),
                    file_path=str(file),
                    content=content,
                )
            )
        return docs

    def load_events(self, industries: list[str] | None = None) -> list[EventRecord]:
        if not self.events_file.exists():
            return []

        industries_set = set(industries) if industries else None

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
            if not industry:
                continue
            if industries_set is not None and industry not in industries_set:
                continue

            event_date = _resolve_event_date(row)
            if event_date is None:
                continue

            report_date = _to_date(row.get("report_date") or row.get("report_date_dt"))
            title = str(row.get("event_name") or row.get("title") or row.get("event_type") or "event").strip()
            content = _resolve_event_content(row)
            confidence = _to_float(row.get("confidence"))

            events.append(
                EventRecord(
                    event_date=event_date,
                    industry=industry,
                    title=title,
                    content=content,
                    report_date=report_date,
                    confidence=confidence,
                )
            )

        events.sort(key=lambda e: e.event_date)
        return events

    def load_market_data(self, industries: list[str]) -> dict[str, pd.DataFrame]:
        if not self.market_dir.exists():
            raise FileNotFoundError(f"Missing market directory: {self.market_dir}")

        code_to_file = _index_bk_files(self.market_dir)
        mapping = self._load_industry_mapping()

        market_data: dict[str, pd.DataFrame] = {}
        self._last_market_files = {}
        for industry in industries:
            file_path = self._resolve_industry_market_file(industry, mapping, code_to_file)
            if file_path is None:
                continue

            df = pd.read_csv(file_path)
            required_cols = {"date", "close"}
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f"Market file {file_path} must contain columns: {sorted(required_cols)}")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            if "ret" in df.columns and df["ret"].notna().any():
                df["ret_1d"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
            else:
                df["ret_1d"] = df["close"].pct_change().fillna(0.0)
            market_data[industry] = df
            self._last_market_files[industry] = str(file_path)
        if not market_data:
            raise FileNotFoundError(
                f"No market files matched for selected industries. "
                f"Check industry_match_file and files under {self.market_dir}"
            )
        return market_data

    def load_market_index(self) -> pd.DataFrame | None:
        if not self.market_index_file or not self.market_index_file.exists():
            return None
        df = pd.read_csv(self.market_index_file)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if "ret" in df.columns and df["ret"].notna().any():
            df["ret_1d"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
        elif "close" in df.columns:
            df["ret_1d"] = df["close"].pct_change().fillna(0.0)
        else:
            df["ret_1d"] = 0.0
        return df

    def _load_industry_mapping(self) -> dict[str, list[dict[str, str]]]:
        if not self.industry_match_file or not self.industry_match_file.exists():
            return {}

        df = pd.read_csv(self.industry_match_file)
        cols = set(df.columns)

        # New format: industry_concept_match_report.csv
        # Priority: exact industry-board match -> exact concept-board match.
        new_required = {"industry", "ind_status", "ind_bk_code", "con_status", "con_bk_code"}
        if new_required.issubset(cols):
            out_new: dict[str, list[dict[str, str]]] = {}
            for _, row in df.iterrows():
                industry = str(row.get("industry", "")).strip()
                if not industry:
                    continue

                picks: list[dict[str, str]] = []
                ind_status = str(row.get("ind_status", "")).strip().lower()
                ind_bk_code = str(row.get("ind_bk_code", "")).strip().upper()
                ind_bk_name = str(row.get("ind_bk_name", "")).strip()
                if ind_status == "exact" and ind_bk_code:
                    picks.append({"bk_code": ind_bk_code, "bk_name": ind_bk_name, "source": "industry"})

                con_status = str(row.get("con_status", "")).strip().lower()
                con_bk_code = str(row.get("con_bk_code", "")).strip().upper()
                con_bk_name = str(row.get("con_bk_name", "")).strip()
                if con_status == "exact" and con_bk_code and con_bk_code != ind_bk_code:
                    picks.append({"bk_code": con_bk_code, "bk_name": con_bk_name, "source": "concept"})

                if picks:
                    out_new[industry] = picks
            return out_new

        # Backward compatibility: industry_match_test_report.csv
        old_required = {"industry", "bk_code", "score"}
        if old_required.issubset(cols):
            out_old: dict[str, list[dict[str, str]]] = {}
            for _, row in df.iterrows():
                industry = str(row.get("industry", "")).strip()
                bk_code = str(row.get("bk_code", "")).strip().upper()
                bk_name = str(row.get("bk_name", "")).strip()
                score = _to_float(row.get("score")) or 0.0
                if not industry or not bk_code or score < 0.5:
                    continue
                out_old.setdefault(industry, []).append({"bk_code": bk_code, "bk_name": bk_name, "source": "legacy"})
            return out_old

        return {}

    def _resolve_industry_market_file(
        self,
        industry: str,
        mapping: dict[str, list[dict[str, str]]],
        code_to_file: dict[str, Path],
    ) -> Path | None:
        # 1) direct name pattern: 行业__BKxxxx.csv
        direct = sorted(self.market_dir.glob(f"{industry}__BK*.csv"))
        if direct:
            return direct[0]

        # 2) via industry-match map
        for item in mapping.get(industry, []):
            bk_code = str(item["bk_code"]).upper()
            if bk_code in code_to_file:
                return code_to_file[bk_code]

        # 3) fallback: 行业.csv
        fallback = self.market_dir / f"{industry}.csv"
        if fallback.exists():
            return fallback
        return None

    def _resolve_report_source_dir(self) -> Path | None:
        if self.report_summary_dir and self.report_summary_dir.exists():
            return self.report_summary_dir
        if self.report_raw_dir and self.report_raw_dir.exists():
            return self.report_raw_dir
        return None


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
        rows.append({"date": parts[0], "industry": parts[1], "title": parts[2], "content": parts[3]})
    return rows


def _to_date(v) -> date | None:
    if v is None:
        return None
    dt = pd.to_datetime(str(v), errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def _to_float(v) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve_event_date(row: dict) -> date | None:
    for key in ("event_date", "event_date_dt", "date", "report_date", "report_date_dt"):
        dt = _to_date(row.get(key))
        if dt is not None:
            return dt
    return None


def _resolve_event_content(row: dict) -> str:
    evidence = str(row.get("evidence_quote") or "").strip()
    if evidence:
        return evidence

    parts = []
    for key in ("actors", "action", "object", "location", "event_type"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    if parts:
        return "; ".join(parts)

    content = str(row.get("content") or "").strip()
    return content


def _index_bk_files(market_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in market_dir.glob("*.csv"):
        m = BK_FILE_PATTERN.search(p.name)
        if not m:
            continue
        code = m.group("code").upper()
        out[code] = p
    return out
