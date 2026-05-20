"""Parity sanity-check against the ``empyrical`` library.

This is a parity sanity-check, NOT a correctness oracle. ``empyrical`` itself
may diverge from textbook conventions on edge cases (e.g., its
``downside_risk`` divisor, its annualization choices, its VaR interpolation).
Where conventions match, the numbers should be close; where they don't,
``riskmetrics`` follows the conventions documented in its own modules and
``empyrical`` is treated as an informational reference only.

The whole module is skipped if ``empyrical`` is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

empyrical = pytest.importorskip("empyrical")

from riskmetrics.drawdown import max_drawdown
from riskmetrics.ratios import omega_ratio, sharpe_ratio
from riskmetrics.returns import cagr
from riskmetrics.tail import value_at_risk


@pytest.fixture(scope="module")
def spy_like_returns() -> pd.Series:
    """SPY-like 1000-day synthetic GBM daily returns (deterministic seed)."""
    rng = np.random.default_rng(20240520)
    mu = 0.0004
    sigma = 0.011
    n = 1000
    samples = rng.normal(mu, sigma, n)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.Series(samples, index=idx, name="spy_like")


def test_sharpe_vs_empyrical(spy_like_returns: pd.Series) -> None:
    ours = sharpe_ratio(spy_like_returns, risk_free=0.0, periods_per_year=252)
    theirs = empyrical.sharpe_ratio(spy_like_returns, risk_free=0.0)
    assert float(ours) == pytest.approx(float(theirs), rel=1e-6)


def test_max_drawdown_vs_empyrical(spy_like_returns: pd.Series) -> None:
    ours = max_drawdown(spy_like_returns)
    theirs = empyrical.max_drawdown(spy_like_returns)
    assert float(ours) == pytest.approx(float(theirs), rel=1e-10, abs=1e-12)


def test_omega_ratio_vs_empyrical(spy_like_returns: pd.Series) -> None:
    ours = omega_ratio(spy_like_returns, threshold=0.0)
    theirs = empyrical.omega_ratio(spy_like_returns, risk_free=0.0, required_return=0.0)
    assert float(ours) == pytest.approx(float(theirs), rel=1e-6)


def test_value_at_risk_vs_empyrical(spy_like_returns: pd.Series) -> None:
    # Our historical VaR uses np.quantile(method="lower") to return an actual
    # observed loss rather than interpolating between adjacent days; empyrical
    # uses default linear interpolation. The two agree to within one sample
    # spacing, which is several basis points on daily equity-like returns.
    ours = value_at_risk(spy_like_returns, confidence=0.95, method="historical")
    theirs = empyrical.value_at_risk(spy_like_returns, cutoff=0.05)
    assert float(ours) == pytest.approx(float(theirs), abs=5e-4)


def test_cagr_vs_empyrical_annual_return(spy_like_returns: pd.Series) -> None:
    ours = cagr(spy_like_returns, periods_per_year=252)
    theirs = empyrical.annual_return(spy_like_returns)
    assert float(ours) == pytest.approx(float(theirs), rel=1e-6)
