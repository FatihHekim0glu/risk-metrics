"""Coverage for return helpers not exercised by the hand-calc suite:
log returns, the CAGR non-positive-wealth guard, calendar-year extrema, and the
monthly returns pivot table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.returns import (
    best_year,
    cagr,
    cumulative_returns,
    log_returns,
    monthly_returns_table,
    worst_year,
)


def test_log_returns_handcalc() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    out = log_returns(prices)
    assert out.iloc[0] == pytest.approx(np.log(110.0 / 100.0), abs=1e-12)
    assert out.iloc[1] == pytest.approx(np.log(99.0 / 110.0), abs=1e-12)


def test_log_returns_sum_equals_log_total_return() -> None:
    prices = pd.Series([100.0, 101.0, 98.98, 100.46, 100.96])
    total_log = float(log_returns(prices).sum())
    assert total_log == pytest.approx(np.log(prices.iloc[-1] / prices.iloc[0]), abs=1e-12)


def test_log_returns_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        log_returns(pd.Series([100.0]))


def test_cagr_raises_on_non_positive_wealth() -> None:
    # A -100% period drives cumulative wealth to exactly zero.
    bad = pd.Series([0.10, -1.0, 0.05])
    with pytest.raises(ValueError, match="non-positive"):
        cagr(bad)


def test_cumulative_returns_matches_manual_compounding() -> None:
    r = pd.Series([0.01, -0.02, 0.015])
    out = cumulative_returns(r)
    assert out.iloc[-1] == pytest.approx((1.01 * 0.98 * 1.015) - 1.0, abs=1e-12)


def test_best_and_worst_year() -> None:
    idx = pd.date_range("2022-01-31", periods=24, freq="ME")
    # 2022 flat-ish, 2023 strongly up.
    vals = [0.0] * 12 + [0.05] * 12
    r = pd.Series(vals, index=idx)
    by, bv = best_year(r)
    wy, wv = worst_year(r)
    assert by == 2023
    assert wy == 2022
    assert bv > wv


def test_best_year_requires_datetimeindex() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        best_year(pd.Series([0.01, 0.02, 0.03]))


def test_worst_year_requires_datetimeindex() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        worst_year(pd.Series([0.01, 0.02, 0.03]))


def test_monthly_returns_table_shape_and_ytd() -> None:
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    r = pd.Series([0.01] * 12, index=idx)
    table = monthly_returns_table(r)
    assert table.index.tolist() == [2023]
    # 12 month columns plus the trailing YTD column.
    assert list(table.columns) == [*range(1, 13), "YTD"]
    # YTD compounds the twelve monthly returns.
    assert table.loc[2023, "YTD"] == pytest.approx(1.01**12 - 1.0, abs=1e-9)


def test_monthly_returns_table_requires_datetimeindex() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        monthly_returns_table(pd.Series([0.01, 0.02]))
