from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.returns import (
    annualized_return_arithmetic,
    best_period,
    cagr,
    cumulative_returns,
    hit_rate,
    simple_returns,
    total_return,
    worst_period,
)


def test_simple_returns_handcalc() -> None:
    prices = pd.Series(
        [100.0, 101.0, 99.0, 100.5, 101.0, 100.0],
        index=pd.date_range("2024-01-02", periods=6, freq="B"),
    )
    expected = pd.Series(
        [0.01, -0.0198019802, 0.0151515152, 0.0049751244, -0.0099009901],
        index=prices.index[1:],
    )
    out = simple_returns(prices)
    assert out.shape == expected.shape
    for got, want in zip(out.to_numpy(), expected.to_numpy()):
        assert got == pytest.approx(want, abs=1e-6)


def test_total_return_tiny(tiny_returns: pd.Series) -> None:
    expected = (1.01) * (0.98) * (1.015) * (1.005) * (0.99) - 1.0
    assert total_return(tiny_returns) == pytest.approx(expected, abs=1e-7)
    # The spec calls out ~0.0001435; double-check that constant too.
    assert total_return(tiny_returns) == pytest.approx(0.0001435, abs=1e-6)


def test_cagr_one_year() -> None:
    # 252 days of constant return r s.t. (1+r)^252 = 1.10 -> total_return == 0.10
    n = 252
    r = (1.10) ** (1.0 / n) - 1.0
    series = pd.Series([r] * n, index=pd.date_range("2024-01-02", periods=n, freq="B"))
    assert cagr(series, periods_per_year=252) == pytest.approx(0.10, abs=1e-9)


def test_annualized_arithmetic_overstates_cagr_under_volatility() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    samples = rng.normal(0.0005, 0.02, n)
    series = pd.Series(samples, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    arith = annualized_return_arithmetic(series, periods_per_year=252)
    geom = cagr(series, periods_per_year=252)
    assert arith > geom


def test_hit_rate(tiny_returns: pd.Series) -> None:
    # 3 of 5 strictly positive in [0.01, -0.02, 0.015, 0.005, -0.01]
    assert hit_rate(tiny_returns, threshold=0.0) == pytest.approx(0.6, abs=1e-12)


def test_best_worst_period(tiny_returns: pd.Series) -> None:
    best_t, best_v = best_period(tiny_returns)
    worst_t, worst_v = worst_period(tiny_returns)
    # tiny_returns index positions: 0:+0.01, 1:-0.02, 2:+0.015, 3:+0.005, 4:-0.01
    assert best_t == tiny_returns.index[2]
    assert best_v == pytest.approx(0.015, abs=1e-12)
    assert worst_t == tiny_returns.index[1]
    assert worst_v == pytest.approx(-0.02, abs=1e-12)


def test_cumulative_returns_guards_against_total_loss() -> None:
    bad = pd.Series([0.5, -1.5])
    with pytest.raises(ValueError):
        cumulative_returns(bad)
