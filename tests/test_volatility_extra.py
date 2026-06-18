"""Coverage for volatility helpers beyond the hand-calc cases:
semi-deviation, mean absolute deviation, rolling-window guards, Ljung-Box."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.volatility import (
    downside_deviation,
    ljung_box_test,
    mean_absolute_deviation,
    realized_volatility,
    rolling_volatility,
    semi_deviation,
)


def test_semi_deviation_equals_downside_deviation_at_mean() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    mean = float(r.mean())
    expected = downside_deviation(r, mar=mean, annualize=False)
    assert semi_deviation(r, annualize=False) == pytest.approx(expected, abs=1e-12)


def test_mean_absolute_deviation_scales_linearly() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    per_period = mean_absolute_deviation(r, annualize=False)
    annual = mean_absolute_deviation(r, periods_per_year=252, annualize=True)
    # MAD annualises linearly, not by sqrt of time.
    assert annual == pytest.approx(per_period * 252, abs=1e-12)


def test_mean_absolute_deviation_handcalc() -> None:
    r = pd.Series([1.0, 3.0, 5.0])
    # mean = 3, deviations |{-2, 0, 2}| -> mean 4/3.
    assert mean_absolute_deviation(r, annualize=False) == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_realized_volatility_without_annualization() -> None:
    r = pd.Series([0.01, -0.01, 0.02, -0.02])
    plain = realized_volatility(r, annualize=False)
    annual = realized_volatility(r, periods_per_year=252, annualize=True)
    assert annual == pytest.approx(plain * np.sqrt(252), abs=1e-12)


def test_rolling_volatility_rejects_small_window() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    with pytest.raises(ValueError, match="window must be an integer"):
        rolling_volatility(r, window=1)


def test_rolling_volatility_full_window_only() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    out = rolling_volatility(r, window=3, annualize=False)
    # First two positions lack a full window, so they are NaN.
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(float(np.std(r.iloc[:3], ddof=1)), abs=1e-12)


def test_ljung_box_returns_finite_tuple() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 200))
    stat, p = ljung_box_test(r, lags=5)
    assert np.isfinite(stat)
    assert 0.0 <= p <= 1.0


def test_ljung_box_detects_strong_autocorrelation() -> None:
    # A near-perfect AR(1) should reject the no-autocorrelation null.
    n = 300
    x = np.zeros(n)
    rng = np.random.default_rng(1)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0.0, 0.001)
    _, p = ljung_box_test(pd.Series(x), lags=5)
    assert p < 0.01
