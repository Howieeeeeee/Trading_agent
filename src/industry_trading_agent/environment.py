from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class StepResult:
    date: date
    nav_before: float
    nav_after: float
    pnl: float
    turnover: float
    transaction_cost: float
    executed_weights: dict[str, float]


class TradingEnvironment:
    def __init__(
        self,
        market_data: dict[str, pd.DataFrame],
        industries: list[str],
        initial_cash: float,
        max_position: float,
        transaction_cost_bps: float,
    ):
        self.market_data = market_data
        self.industries = industries
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_cost_rate = transaction_cost_bps / 10_000.0

        self.current_weights = {industry: 0.0 for industry in industries}
        self.nav = initial_cash
        self.nav_history = [initial_cash]

    def step(self, dt: date, target_weights: dict[str, float]) -> StepResult:
        sanitized = self._sanitize_weights(target_weights)

        nav_before = self.nav
        turnover = sum(abs(sanitized[i] - self.current_weights.get(i, 0.0)) for i in self.industries)
        transaction_cost = nav_before * turnover * self.transaction_cost_rate

        portfolio_return = 0.0
        for industry in self.industries:
            ret = self._get_industry_return(industry, dt)
            portfolio_return += sanitized[industry] * ret

        pnl = nav_before * portfolio_return - transaction_cost
        nav_after = nav_before + pnl

        self.current_weights = sanitized
        self.nav = nav_after
        self.nav_history.append(nav_after)

        return StepResult(
            date=dt,
            nav_before=nav_before,
            nav_after=nav_after,
            pnl=pnl,
            turnover=turnover,
            transaction_cost=transaction_cost,
            executed_weights=sanitized,
        )

    def _sanitize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        sanitized = {industry: max(0.0, float(weights.get(industry, 0.0))) for industry in self.industries}

        total = sum(sanitized.values())
        if total <= 1e-12:
            return {industry: 0.0 for industry in self.industries}

        normalized = {industry: v / total for industry, v in sanitized.items()}
        capped = {industry: min(self.max_position, normalized[industry]) for industry in self.industries}

        capped_total = sum(capped.values())
        if capped_total > 1.0:
            return {industry: v / capped_total for industry, v in capped.items()}
        return capped

    def _get_industry_return(self, industry: str, dt: date) -> float:
        df = self.market_data[industry]
        row = df[df["date"] == pd.Timestamp(dt)]
        if row.empty:
            return 0.0
        return float(row.iloc[0]["ret_1d"])
