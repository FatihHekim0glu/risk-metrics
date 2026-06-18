"""Coverage for risk-adjusted ratios beyond the hand-calc suite:
risk-free Series handling, degenerate denominators, Treynor, and M-squared."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.ratios import (
    m_squared,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)


def test_sharpe_accepts_risk_free_series() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015], index=idx)
    rf_series = pd.Series(0.02, index=idx)
    scalar = sharpe_ratio(r, risk_free=0.02, periods_per_year=252)
    series = sharpe_ratio(r, risk_free=rf_series, periods_per_year=252)
    # A flat annual-rate Series must match the scalar form.
    assert series == pytest.approx(scalar, abs=1e-9)


def test_sharpe_smart_matches_naive_when_no_autocorrelation() -> None:
    # An almost-iid series should leave the Lo adjustment close to 1.
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    naive = sharpe_ratio(r, periods_per_year=252, smart=False)
    smart = sharpe_ratio(r, periods_per_year=252, smart=True)
    assert smart == pytest.approx(naive, rel=0.25)


def test_sortino_nan_when_no_downside() -> None:
    # All returns at or above the MAR -> zero downside deviation.
    r = pd.Series([0.01, 0.02, 0.03, 0.015])
    with pytest.warns(UserWarning, match="zero downside"):
        out = sortino_ratio(r, mar=0.0)
    assert np.isnan(out)


def test_omega_ratio_infinite_when_no_losses() -> None:
    r = pd.Series([0.01, 0.02, 0.03])
    assert omega_ratio(r, threshold=0.0) == float("inf")


def test_omega_ratio_handcalc() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    # gains above 0 = 0.05, losses below 0 = 0.03 -> 5/3.
    assert omega_ratio(r, threshold=0.0) == pytest.approx(5.0 / 3.0, abs=1e-12)


def test_treynor_ratio_divides_by_beta() -> None:
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    t1 = treynor_ratio(r, beta=1.0, periods_per_year=252)
    t2 = treynor_ratio(r, beta=2.0, periods_per_year=252)
    assert t1 == pytest.approx(2.0 * t2, abs=1e-9)


def test_treynor_ratio_nan_on_zero_beta() -> None:
    r = pd.Series([0.01, -0.005, 0.02])
    with pytest.warns(UserWarning, match="beta is zero"):
        out = treynor_ratio(r, beta=0.0)
    assert np.isnan(out)


def test_treynor_ratio_nan_on_nan_beta() -> None:
    r = pd.Series([0.01, -0.005, 0.02])
    with pytest.warns(UserWarning, match="beta is zero or NaN"):
        out = treynor_ratio(r, beta=float("nan"))
    assert np.isnan(out)


def test_m_squared_equals_sharpe_times_vol_plus_rf() -> None:
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    sr = sharpe_ratio(r, risk_free=0.0, periods_per_year=252)
    expected = sr * 0.16 + 0.0
    assert m_squared(r, benchmark_vol_annualized=0.16) == pytest.approx(expected, abs=1e-9)


def test_sharpe_raises_on_non_overlapping_risk_free_series() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    r = pd.Series([0.01, -0.005, 0.02, 0.0], index=idx)
    # Risk-free Series indexed entirely before the returns window.
    rf = pd.Series(0.02, index=pd.date_range("2010-01-01", periods=4, freq="B"))
    with pytest.raises(ValueError, match="no overlapping observations"):
        sharpe_ratio(r, risk_free=rf)


def test_m_squared_with_risk_free_series() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015], index=idx)
    rf_series = pd.Series(0.03, index=idx)
    out = m_squared(r, benchmark_vol_annualized=0.16, risk_free=rf_series)
    # The annual risk-free average (0.03) is added back into the M-squared.
    assert out > m_squared(r, benchmark_vol_annualized=0.16, risk_free=0.0)
