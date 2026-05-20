from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from riskmetrics.volatility import (
    autocorrelation,
    downside_deviation,
    realized_volatility,
    rolling_volatility,
)


def test_realized_vol_handcalc(tiny_returns: pd.Series) -> None:
    # std with ddof=1 of [0.01, -0.02, 0.015, 0.005, -0.01]
    sample_std = tiny_returns.std(ddof=1)
    assert sample_std == pytest.approx(0.013509256, abs=1e-6)
    annualized = sample_std * math.sqrt(252)
    out = realized_volatility(tiny_returns, periods_per_year=252)
    assert out == pytest.approx(annualized, abs=1e-9)
    assert out == pytest.approx(0.21443, abs=1e-4)


def test_downside_deviation_divides_by_N(tiny_returns: pd.Series) -> None:
    # MAR=0: downside excursions are -0.02 and -0.01.
    # If divisor is N (=5):  sqrt((0.0004 + 0.0001)/5) = sqrt(0.0001) = 0.01
    # If divisor is N_down (=2): sqrt((0.0004 + 0.0001)/2) ~ 0.01581
    # Per spec: must equal 0.01 -> proves N divisor.
    out = downside_deviation(tiny_returns, mar=0.0, periods_per_year=1)
    assert out == pytest.approx(0.01, abs=1e-12)


def test_rolling_volatility_min_periods_window() -> None:
    rng = np.random.default_rng(11)
    n = 60
    series = pd.Series(
        rng.normal(0.0, 0.01, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    window = 20
    out = rolling_volatility(series, window=window, periods_per_year=252)
    # The first window-1 entries should be NaN (no peeking, full window required)
    assert out.iloc[: window - 1].isna().all()
    # And the window-th value onwards should be finite.
    assert out.iloc[window - 1 :].notna().all()


def test_autocorrelation_lag1() -> None:
    rng = np.random.default_rng(2024)
    n = 5000
    rho = 0.5
    eps = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    series = pd.Series(x, index=pd.date_range("2000-01-03", periods=n, freq="B"))
    est = autocorrelation(series, lag=1)
    assert 0.4 <= est <= 0.6
