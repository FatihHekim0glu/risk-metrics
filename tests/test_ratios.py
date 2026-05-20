from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from riskmetrics.ratios import omega_ratio, sharpe_ratio, sortino_ratio


def test_sharpe_tiny(tiny_returns: pd.Series) -> None:
    # mean of [0.01, -0.02, 0.015, 0.005, -0.01] is exactly 0; Sharpe = 0/vol = 0.
    val = sharpe_ratio(tiny_returns, risk_free=0.0, periods_per_year=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_sharpe_annualization_factor() -> None:
    rng = np.random.default_rng(99)
    n = 1500
    series = pd.Series(
        rng.normal(0.0005, 0.01, n),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    sa = sharpe_ratio(series, risk_free=0.0, periods_per_year=252)
    s1 = sharpe_ratio(series, risk_free=0.0, periods_per_year=1)
    ratio = sa / s1
    assert ratio == pytest.approx(math.sqrt(252), rel=1e-3)


def test_sortino_divides_by_total_N(tiny_returns: pd.Series) -> None:
    # mean of tiny is exactly 0, so Sortino = 0/downside_dev = 0.
    # The N-divisor property is exercised separately in test_volatility.
    val = sortino_ratio(tiny_returns, mar=0.0, periods_per_year=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_sharpe_smart_lower_than_naive_under_positive_autocorrelation() -> None:
    rng = np.random.default_rng(17)
    n = 2000
    rho = 0.3
    eps = rng.normal(0.0005, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    series = pd.Series(
        x, index=pd.date_range("2010-01-01", periods=n, freq="B")
    )
    naive = sharpe_ratio(series, risk_free=0.0, periods_per_year=252, smart=False)
    smart = sharpe_ratio(series, risk_free=0.0, periods_per_year=252, smart=True)
    assert smart < naive


def test_sharpe_zero_vol_returns_nan_with_warning() -> None:
    series = pd.Series([0.001] * 100)
    with pytest.warns(UserWarning, match="zero volatility"):
        val = sharpe_ratio(series, risk_free=0.0, periods_per_year=252)
    assert np.isnan(val)


def test_omega_ratio_threshold_zero(tiny_returns: pd.Series) -> None:
    # gains above 0: 0.01 + 0.015 + 0.005 = 0.030
    # |losses below 0|: 0.02 + 0.01 = 0.030
    # omega = 0.030 / 0.030 = 1.0
    val = omega_ratio(tiny_returns, threshold=0.0)
    assert val == pytest.approx(1.0, abs=1e-12)
