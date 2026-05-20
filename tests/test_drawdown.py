from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.drawdown import (
    calmar_ratio,
    drawdown_series,
    drawdown_table,
    max_drawdown,
    ulcer_index,
)


def test_max_drawdown_tiny(tiny_returns: pd.Series) -> None:
    # Wealth trajectory:
    #   day0: 1.01
    #   day1: 1.01 * 0.98 = 0.9898    <- valley
    #   day2: 0.9898 * 1.015 = 1.00465
    #   ...
    # Running peak at day1 is 1.01; drawdown there = 0.9898/1.01 - 1 = -0.0198
    dd = drawdown_series(tiny_returns)
    assert dd.min() == pytest.approx(-0.0198, abs=1e-4)
    assert max_drawdown(tiny_returns) == pytest.approx(-0.0198, abs=1e-4)


def test_drawdown_table_synthetic(synthetic_drawdown_returns: pd.Series) -> None:
    """Peak at end of +1% run, valley at end of -3% run, recovery in subsequent +1% run."""
    table = drawdown_table(synthetic_drawdown_returns, top=1)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1
    row = table.iloc[0]

    idx = synthetic_drawdown_returns.index
    # Peak at day 9 (last of the first +1% run)
    assert row["peak_date"] == idx[9]
    # Valley at day 19 (last of the -3% run)
    assert row["valley_date"] == idx[19]

    # Recovery: wealth re-attains peak during days 20..39 (+1% run).
    # peak wealth = 1.01^10; valley wealth = 1.01^10 * 0.97^10;
    # we need k s.t. 0.97^10 * 1.01^k >= 1  ->  k >= 10*ln(0.97)/ln(1.01) * -1
    # k_min = ceil(10 * ln(1/0.97) / ln(1.01)) = ceil(3.0612) = 31
    # so recovery date is day 19 + 31 = day 50? Recompute carefully:
    #   ratio_needed = 1 / 0.97^10 = 1.3439...
    #   ln(1.3439)/ln(1.01) = 29.62 -> ceil = 30
    # recovery_date = idx[19 + 30] = idx[49]
    # But this falls inside the flat tail (days 40..69) which has 0% return,
    # so the +1% run from day 20..39 alone can't recover. Check carefully:
    # wealth at day 39 = valley_wealth * 1.01^20 = 0.97^10 * 1.01^10 * 1.01^20
    #                  = 1.01^30 * 0.97^10
    # log10 check: 30*log10(1.01) + 10*log10(0.97)
    #            = 30*0.00432 + 10*(-0.01323) = 0.1296 - 0.1323 = -0.0027
    # so 10^-0.0027 ~= 0.9938 -> day 39 wealth still 0.62% below peak.
    # Then 30 flat days never recover -> drawdown is OPEN at end.
    # The assertion the spec wants ("recovery_date == day 30") cannot be
    # satisfied; treat this drawdown as still open at series end.
    assert bool(row["is_open"]) is True
    assert pd.isna(row["recovery_date"])


def test_drawdown_table_open_at_end() -> None:
    # 5 up days, then 3 down days, ending mid-drawdown.
    daily = [0.01] * 5 + [-0.05] * 3
    idx = pd.date_range("2024-01-02", periods=len(daily), freq="B")
    series = pd.Series(daily, index=idx)
    table = drawdown_table(series, top=1)
    assert len(table) == 1
    row = table.iloc[0]
    assert bool(row["is_open"]) is True
    assert pd.isna(row["recovery_date"])


def test_ulcer_index_positive() -> None:
    rng = np.random.default_rng(7)
    n = 500
    noisy = pd.Series(
        rng.normal(0.0, 0.01, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    assert ulcer_index(noisy) >= 0.0

    # Constant non-negative returns -> wealth is non-decreasing -> no drawdown.
    flat = pd.Series(
        [0.001] * 200, index=pd.date_range("2020-01-01", periods=200, freq="B")
    )
    assert ulcer_index(flat) == pytest.approx(0.0, abs=1e-12)


def test_calmar_ratio_sign() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    # Positive drift but with vol, so a drawdown occurs.
    series = pd.Series(
        rng.normal(0.001, 0.01, n),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    val = calmar_ratio(series, periods_per_year=252)
    assert np.isfinite(val)
    assert val > 0.0
