"""Coverage for tail-risk metrics beyond the hand-calc suite:
parametric VaR/CVaR, method dispatch and validation, skewness, tail ratio,
PSR degenerate paths, and the Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskmetrics.tail import (
    conditional_value_at_risk,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    skewness,
    tail_ratio,
    value_at_risk,
)


def test_value_at_risk_parametric_formula() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    out = value_at_risk(r, confidence=0.95, method="parametric")
    expected = float(r.mean() + r.std(ddof=1) * stats.norm.ppf(0.05))
    assert out == pytest.approx(expected, abs=1e-12)


def test_value_at_risk_gaussian_alias_matches_parametric() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    assert value_at_risk(r, method="gaussian") == pytest.approx(
        value_at_risk(r, method="parametric"), abs=1e-12
    )


def test_value_at_risk_cornish_fisher_dispatch() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    from riskmetrics.tail import cornish_fisher_var

    assert value_at_risk(r, method="cornish_fisher") == pytest.approx(
        cornish_fisher_var(r), abs=1e-12
    )


def test_value_at_risk_unknown_method_raises() -> None:
    r = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError, match="Unknown method"):
        value_at_risk(r, method="bogus")


def test_cvar_parametric_formula() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    out = conditional_value_at_risk(r, confidence=0.95, method="parametric")
    z = stats.norm.ppf(0.05)
    expected = float(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / 0.05)
    assert out == pytest.approx(expected, abs=1e-12)


def test_cvar_unknown_method_raises() -> None:
    r = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError, match="Unknown method"):
        conditional_value_at_risk(r, method="bogus")


def test_skewness_sign_for_left_skewed_data() -> None:
    # One large negative outlier pulls the skew negative.
    r = pd.Series([0.01, 0.01, 0.01, 0.01, -0.20])
    assert skewness(r) < 0.0


def test_tail_ratio_infinite_when_left_quantile_zero() -> None:
    # A right-only tail with the lower quantile pinned to zero -> infinite ratio.
    r = pd.Series([0.0, 0.0, 0.0, 0.0, 0.05, 0.10])
    assert tail_ratio(r, percentile=0.2) == float("inf")


def test_probabilistic_sharpe_nan_on_zero_volatility() -> None:
    r = pd.Series([0.01, 0.01, 0.01, 0.01])
    with pytest.warns(UserWarning, match="zero volatility"):
        out = probabilistic_sharpe_ratio(r)
    assert np.isnan(out)


def test_deflated_sharpe_rejects_too_few_trials() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    with pytest.raises(ValueError, match="n_trials must be >= 2"):
        deflated_sharpe_ratio(r, n_trials=1, sr_variance=0.5)


def test_deflated_sharpe_rejects_negative_variance() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    with pytest.raises(ValueError, match="sr_variance must be non-negative"):
        deflated_sharpe_ratio(r, n_trials=10, sr_variance=-0.1)


def test_deflated_sharpe_below_psr_at_zero() -> None:
    # The selection-bias threshold is strictly positive, so DSR cannot exceed
    # the probability of beating zero.
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    psr0 = probabilistic_sharpe_ratio(r, threshold_sr=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=50, sr_variance=1.0)
    assert 0.0 <= dsr <= 1.0
    assert dsr <= psr0 + 1e-9
