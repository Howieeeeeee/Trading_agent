from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .agent import TradingAgent
from .config import AppConfig
from .data_loader import DataLoader
from .environment import TradingEnvironment
from .providers.llm_factory import build_llm
from .retrieval import ContextRetriever


class BacktestRunner:
    def __init__(self, config: AppConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path) if config_path else None

    def run(self) -> dict:
        mode = _mode_from_config_path(self.config_path)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        loader = DataLoader(
            report_raw_dir=self.config.data.report_raw_dir,
            report_summary_dir=self.config.data.report_summary_dir,
            events_file=self.config.data.events_file,
            market_dir=self.config.data.market_dir,
            industry_match_file=self.config.data.industry_match_file,
            market_index_file=self.config.data.market_index_file,
        )

        all_events = loader.load_events(industries=None) if self.config.trading.use_events else []
        industries = _resolve_industries(self.config, all_events)

        reports = loader.load_reports(industries) if self.config.trading.use_reports else []
        events = [e for e in all_events if e.industry in set(industries)]
        market_data = loader.load_market_data(industries)
        industries = [i for i in industries if i in market_data]
        if not industries:
            raise ValueError("No tradable industries after industry/concept exact matching and market file filtering.")
        reports = [r for r in reports if r.industry in set(industries)]
        events = [e for e in events if e.industry in set(industries)]
        market_index = loader.load_market_index()

        retriever = ContextRetriever(
            reports=reports,
            events=events,
            market_data=market_data,
            market_index=market_index,
        )

        env = TradingEnvironment(
            market_data=market_data,
            industries=industries,
            initial_cash=self.config.trading.initial_cash,
            max_position=self.config.trading.max_position,
            transaction_cost_bps=self.config.trading.transaction_cost_bps,
        )

        llm = build_llm(self.config.llm)
        agent = TradingAgent(llm=llm, trading_cfg=self.config.trading)

        trade_dates = _build_trade_dates(
            market_data,
            industries,
            self.config.trading.start_date,
            self.config.trading.end_date,
            self.config.trading.rebalance_frequency,
        )
        if len(trade_dates) < 2:
            raise ValueError("Not enough trade dates. Need at least 2 dates for T+1 execution.")

        decisions: list[dict] = []
        daily_records: list[dict] = []
        trading_logs: list[dict] = []

        # T day decides position for T+1 day, to avoid look-ahead bias.
        for i in range(len(trade_dates) - 1):
            as_of_date = trade_dates[i]
            exec_date = trade_dates[i + 1]

            context = retriever.build_context(
                as_of_date=as_of_date,
                industries=industries,
                lookback_days=self.config.trading.lookback_days,
            )

            decision = agent.decide(
                as_of_date=as_of_date,
                industries=industries,
                market_summary=context["market_summary"],
                event_summary=context["event_summary"],
                report_summary=context["report_summary"],
                current_positions=env.current_weights,
                nav_history=env.nav_history,
            )

            step = env.step(dt=exec_date, target_weights=decision.weights)

            decisions.append(
                {
                    "decision_date": as_of_date.isoformat(),
                    "execute_date": exec_date.isoformat(),
                    "reasoning": decision.reasoning,
                    "weights": decision.weights,
                }
            )
            daily_records.append(
                {
                    "decision_date": as_of_date.isoformat(),
                    "execute_date": exec_date.isoformat(),
                    "nav_before": step.nav_before,
                    "nav_after": step.nav_after,
                    "pnl": step.pnl,
                    "turnover": step.turnover,
                    "transaction_cost": step.transaction_cost,
                    **{f"w_{k}": v for k, v in step.executed_weights.items()},
                }
            )

            referenced_files = sorted(
                set(_collect_daily_files(context, loader, industries, self.config.data.market_index_file))
            )
            trading_logs.append(
                {
                    "mode": mode,
                    "run_id": run_id,
                    "config_file": str(self.config_path) if self.config_path else None,
                    "decision_date": as_of_date.isoformat(),
                    "execute_date": exec_date.isoformat(),
                    "use_reports": self.config.trading.use_reports,
                    "use_events": self.config.trading.use_events,
                    "industries": industries,
                    "referenced_files": referenced_files,
                    "report_refs": context.get("report_refs", []),
                    "event_refs": context.get("event_refs", []),
                    "market_summary": context["market_summary"],
                    "event_summary": context["event_summary"],
                    "report_summary": context["report_summary"],
                    "decision_weights": decision.weights,
                    "decision_reasoning": decision.reasoning,
                    "execution": {
                        "nav_before": step.nav_before,
                        "nav_after": step.nav_after,
                        "pnl": step.pnl,
                        "turnover": step.turnover,
                        "transaction_cost": step.transaction_cost,
                    },
                }
            )

        result = _build_result_summary(
            initial_nav=self.config.trading.initial_cash,
            final_nav=env.nav,
            daily_records=daily_records,
        )
        result["industries"] = industries
        result["mode"] = mode
        result["run_id"] = run_id

        out_root = self.config.output.output_dir
        out_dir = out_root / f"{mode}__{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        daily_records_path = out_dir / "daily_records.csv"
        decisions_path = out_dir / "decisions.json"
        summary_path = out_dir / "summary.json"
        trading_log_jsonl_path = out_dir / "trading_log.jsonl"
        trading_log_json_path = out_dir / "trading_log.json"

        pd.DataFrame(daily_records).to_csv(daily_records_path, index=False)
        pd.DataFrame(decisions).to_json(decisions_path, orient="records", force_ascii=False, indent=2)
        pd.DataFrame([result]).to_json(summary_path, orient="records", force_ascii=False, indent=2)
        pd.DataFrame(trading_logs).to_json(trading_log_json_path, orient="records", force_ascii=False, indent=2)

        with trading_log_jsonl_path.open("w", encoding="utf-8") as f:
            for item in trading_logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return {
            "summary": result,
            "mode": mode,
            "run_id": run_id,
            "output_dir": str(out_dir),
            "daily_records_path": str(daily_records_path),
            "decisions_path": str(decisions_path),
            "summary_path": str(summary_path),
            "trading_log_json_path": str(trading_log_json_path),
            "trading_log_jsonl_path": str(trading_log_jsonl_path),
        }


def _resolve_industries(config: AppConfig, events) -> list[str]:
    if config.trading.industries:
        return config.trading.industries

    if not config.trading.auto_select_industries:
        raise ValueError("No industries configured. Set trading.industries or enable auto_select_industries.")
    if not config.trading.use_events:
        raise ValueError("auto_select_industries requires use_events=true when industries is empty.")

    start = pd.Timestamp(config.trading.start_date).date()
    end = pd.Timestamp(config.trading.end_date).date()

    counter: Counter[str] = Counter()
    for e in events:
        if start <= e.event_date <= end:
            counter[e.industry] += 1

    selected = [
        industry
        for industry, cnt in counter.most_common(config.trading.auto_select_top_n)
        if cnt >= config.trading.min_event_count
    ]
    if not selected:
        raise ValueError("Auto industry selection found no candidates in the window.")
    return selected


def _build_trade_dates(
    market_data: dict[str, pd.DataFrame],
    industries: list[str],
    start_date: str,
    end_date: str,
    rebalance_frequency: str,
) -> list[date]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    common_dates: set[pd.Timestamp] | None = None
    for industry in industries:
        df = market_data[industry]
        dates = set(df[(df["date"] >= start) & (df["date"] <= end)]["date"].tolist())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    if not common_dates:
        raise ValueError("No common market dates in selected range")

    all_dates = sorted(common_dates)
    if rebalance_frequency.upper() == "D":
        selected = all_dates
    elif rebalance_frequency.upper() == "W":
        selected = [d for i, d in enumerate(all_dates) if i % 5 == 0]
    else:
        raise ValueError("Unsupported rebalance_frequency, use 'D' or 'W'")

    return [d.date() for d in selected]


def _build_result_summary(initial_nav: float, final_nav: float, daily_records: list[dict]) -> dict:
    df = pd.DataFrame(daily_records)
    if df.empty:
        return {
            "initial_nav": initial_nav,
            "final_nav": final_nav,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "days": 0,
        }

    daily_ret = df["nav_after"].pct_change().fillna(0.0)
    total_return = final_nav / initial_nav - 1.0
    days = len(df)
    ann_ret = (1 + total_return) ** (252 / max(days, 1)) - 1 if days > 0 else 0.0
    ann_vol = daily_ret.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    nav_series = df["nav_after"]
    rolling_max = nav_series.cummax()
    dd = (nav_series / rolling_max - 1.0).min()

    return {
        "initial_nav": float(initial_nav),
        "final_nav": float(final_nav),
        "total_return": float(total_return),
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "days": int(days),
    }


def _mode_from_config_path(config_path: Path | None) -> str:
    if not config_path:
        return "manual"
    stem = config_path.stem.strip().lower()
    if not stem:
        return "manual"
    return stem


def _collect_daily_files(context: dict, loader: DataLoader, industries: list[str], market_index_file: Path | None) -> list[str]:
    files: list[str] = []
    for ref in context.get("report_refs", []):
        file_path = ref.get("file_path")
        if file_path:
            files.append(str(file_path))

    if context.get("event_refs"):
        files.append(str(loader.events_file))

    for industry in industries:
        market_file = loader._last_market_files.get(industry)
        if market_file:
            files.append(str(market_file))

    if market_index_file:
        files.append(str(market_index_file))
    return files
