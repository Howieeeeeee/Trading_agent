from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1200
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    extra: dict[str, Any] | None = None


@dataclass
class DataConfig:
    report_raw_dir: Path | None
    report_summary_dir: Path | None
    events_file: Path
    market_dir: Path
    industry_match_file: Path | None = None
    market_index_file: Path | None = None


@dataclass
class TradingConfig:
    industries: list[str]
    start_date: str
    end_date: str
    lookback_days: int = 20
    initial_cash: float = 1_000_000.0
    max_position: float = 0.35
    transaction_cost_bps: float = 10.0
    rebalance_frequency: str = "D"
    auto_select_industries: bool = False
    auto_select_top_n: int = 6
    min_event_count: int = 1


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs")
    save_daily_positions: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig
    data: DataConfig
    trading: TradingConfig
    output: OutputConfig



def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]



def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    llm_raw = raw.get("llm", {})
    data_raw = _require(raw, "data")
    trading_raw = _require(raw, "trading")
    output_raw = raw.get("output", {})

    app = AppConfig(
        llm=LLMConfig(
            provider=llm_raw.get("provider", "openai"),
            model=llm_raw.get("model", "gpt-4o-mini"),
            temperature=float(llm_raw.get("temperature", 0.0)),
            max_tokens=int(llm_raw.get("max_tokens", 1200)),
            base_url=llm_raw.get("base_url"),
            api_key_env=llm_raw.get("api_key_env", "OPENAI_API_KEY"),
            extra=llm_raw.get("extra"),
        ),
        data=DataConfig(
            report_raw_dir=Path(data_raw["report_raw_dir"]) if data_raw.get("report_raw_dir") else None,
            report_summary_dir=Path(data_raw["report_summary_dir"]) if data_raw.get("report_summary_dir") else None,
            events_file=Path(_require(data_raw, "events_file")),
            market_dir=Path(_require(data_raw, "market_dir")),
            industry_match_file=Path(data_raw["industry_match_file"]) if data_raw.get("industry_match_file") else None,
            market_index_file=Path(data_raw["market_index_file"]) if data_raw.get("market_index_file") else None,
        ),
        trading=TradingConfig(
            industries=list(trading_raw.get("industries", [])),
            start_date=str(_require(trading_raw, "start_date")),
            end_date=str(_require(trading_raw, "end_date")),
            lookback_days=int(trading_raw.get("lookback_days", 20)),
            initial_cash=float(trading_raw.get("initial_cash", 1_000_000.0)),
            max_position=float(trading_raw.get("max_position", 0.35)),
            transaction_cost_bps=float(trading_raw.get("transaction_cost_bps", 10.0)),
            rebalance_frequency=str(trading_raw.get("rebalance_frequency", "D")),
            auto_select_industries=bool(trading_raw.get("auto_select_industries", False)),
            auto_select_top_n=int(trading_raw.get("auto_select_top_n", 6)),
            min_event_count=int(trading_raw.get("min_event_count", 1)),
        ),
        output=OutputConfig(
            output_dir=Path(output_raw.get("output_dir", "outputs")),
            save_daily_positions=bool(output_raw.get("save_daily_positions", True)),
        ),
    )
    return app
