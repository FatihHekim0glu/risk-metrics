from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_returns() -> pd.Series:
    """Canonical 5-element hand-calculable return series."""
    return pd.Series(
        [0.01, -0.02, 0.015, 0.005, -0.01],
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
        name="tiny",
    )


@pytest.fixture(params=[0, 1, 7, 42, 2024])
def gbm_returns(request) -> pd.Series:
    """1000-step Geometric Brownian Motion daily returns, parameterized over seeds."""
    seed = request.param
    rng = np.random.default_rng(seed)
    mu = 0.0005
    sigma = 0.01
    n = 1000
    samples = rng.normal(loc=mu, scale=sigma, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(samples, index=idx, name=f"gbm_seed{seed}")


@pytest.fixture
def synthetic_drawdown_returns() -> pd.Series:
    """A returns series engineered for an obvious peak->valley->recovery pattern.

    Structure (70 business days total):
      - Days 0..9   (10 days): +1.0% each   -> wealth rises, peak hit at day 9
      - Days 10..19 (10 days): -3.0% each   -> drawdown, valley at day 19
      - Days 20..39 (20 days): +1.0% each   -> recovery
      - Days 40..69 (30 days): 0.0%         -> flat tail

    Hand-check:
      cumulative wealth at day 9  = 1.01^10        ~= 1.1046
      cumulative wealth at day 19 = 1.01^10 * 0.97^10 ~= 0.8147
      drawdown at day 19          = 0.8147/1.1046 - 1 ~= -0.2624
      wealth recovers past the day-9 peak somewhere in the +1% stretch;
      with 20 days of +1.0% growth from valley, day 30 wealth ~= 0.9938 (not yet)
      day 31 ~= 1.0037 -> first day strictly above the prior peak.
    The exact recovery date is asserted approximately in the test.
    """
    daily = (
        [0.01] * 10  # days 0..9   (peak builds)
        + [-0.03] * 10  # days 10..19 (valley)
        + [0.01] * 20  # days 20..39 (recovery + new highs)
        + [0.0] * 30  # days 40..69 (flat)
    )
    idx = pd.date_range("2024-01-02", periods=len(daily), freq="B")
    return pd.Series(daily, index=idx, name="synthetic_dd")


@pytest.fixture
def benchmark_pair() -> tuple[pd.Series, pd.Series]:
    """Two correlated daily return series (asset, bench)."""
    rng = np.random.default_rng(123)
    n = 750
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0004, 0.009, n), index=idx, name="bench")
    noise = pd.Series(rng.normal(0.0, 0.004, n), index=idx, name="noise")
    asset = (1.2 * bench + 0.0002 + noise).rename("asset")
    return asset, bench


@pytest.fixture
def rf_series(tiny_returns: pd.Series) -> pd.Series:
    """Flat 5% annual risk-free rate aligned to tiny_returns index."""
    return pd.Series(0.05, index=tiny_returns.index, name="rf")
