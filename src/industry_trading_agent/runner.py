from __future__ import annotations

from datetime import date

import pandas as pd

from .agent import TradingAgent
from .config import AppConfig
from .data_loader import DataLoader
from .environment import TradingEnvironment
from .providers.llm_factory import build_llm
from .retrieval import ContextRetriever


class BacktestRunner:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> dict:
        loaded = DataLoader(
            reports_dir=self.config.data.reports_dir,
            events_file=self.config.data.events_file,
            market_dir=self.config.data.market_dir,
        ).load_all(self.config.trading.industries)

        retriever = ContextRetriever(
            reports=loaded.reports,
            events=loaded.events,
            market_data=loaded.market_data,
        )

        env = TradingEnvironment(
            market_data=loaded.market_data,
            industries=self.config.trading.industries,
            initial_cash=self.config.trading.initial_cash,
            max_position=self.config.trading.max_position,
            transaction_cost_bps=self.config.trading.transaction_cost_bps,
        )

        llm = build_llm(self.config.llm)
        agent = TradingAgent(llm=llm, trading_cfg=self.config.trading)

        trade_dates = _build_trade_dates(
            loaded.market_data,
            self.config.trading.industries,
            self.config.trading.start_date,
            self.config.trading.end_date,
            self.config.trading.rebalance_frequency,
        )

        decisions: list[dict] = []
        daily_records: list[dict] = []

        for dt in trade_dates:
            context = retriever.build_context(
                as_of_date=dt,
                industries=self.config.trading.industries,
                lookback_days=self.config.trading.lookback_days,
            )

            decision = agent.decide(
                as_of_date=dt,
                industries=self.config.trading.industries,
                market_summary=context["market_summary"],
                event_summary=context["event_summary"],
                report_summary=context["report_summary"],
                current_positions=env.current_weights,
                nav_history=env.nav_history,
            )

            step = env.step(dt=dt, target_weights=decision.weights)

            decisions.append(
                {
                    "date": dt.isoformat(),
                    "reasoning": decision.reasoning,
                    "weights": decision.weights,
                }
            )
            daily_records.append(
                {
                    "date": dt.isoformat(),
                    "nav_before": step.nav_before,
                    "nav_after": step.nav_after,
                    "pnl": step.pnl,
                    "turnover": step.turnover,
                    "transaction_cost": step.transaction_cost,
                    **{f"w_{k}": v for k, v in step.executed_weights.items()},
                }
            )

        result = _build_result_summary(
            initial_nav=self.config.trading.initial_cash,
            final_nav=env.nav,
            daily_records=daily_records,
        )

        out_dir = self.config.output.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(daily_records).to_csv(out_dir / "daily_records.csv", index=False)
        pd.DataFrame(decisions).to_json(out_dir / "decisions.json", orient="records", force_ascii=False, indent=2)
        pd.DataFrame([result]).to_json(out_dir / "summary.json", orient="records", force_ascii=False, indent=2)

        return {
            "summary": result,
            "daily_records_path": str(out_dir / "daily_records.csv"),
            "decisions_path": str(out_dir / "decisions.json"),
            "summary_path": str(out_dir / "summary.json"),
        }


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
