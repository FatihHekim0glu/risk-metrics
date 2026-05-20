from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck
from hypothesis import settings as hyp_settings
from hypothesis import strategies as hyp_st
from hypothesis import given

from riskmetrics.drawdown import max_drawdown
from riskmetrics.ratios import sharpe_ratio
from riskmetrics.returns import cumulative_returns
from riskmetrics.tail import (
    conditional_value_at_risk,
    probabilistic_sharpe_ratio,
    value_at_risk,
)
from riskmetrics.volatility import realized_volatility


returns_strategy = hyp_st.lists(
    hyp_st.floats(
        min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False, width=64
    ),
    min_size=30,
    max_size=2000,
).map(lambda xs: pd.Series(xs))


_SETTINGS = hyp_settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_SETTINGS
@given(r=returns_strategy)
def test_property_max_drawdown_in_bounds(r: pd.Series) -> None:
    md = float(max_drawdown(r))
    # Max drawdown is in (-1, 0]. Allow tiny floating slack at the bounds.
    assert -1.0 - 1e-12 <= md <= 0.0 + 1e-12


@_SETTINGS
@given(r=returns_strategy)
def test_property_sharpe_scale_invariant_under_positive_scaling(r: pd.Series) -> None:
    # Skip degenerate (zero-vol) samples; their Sharpe is NaN by convention.
    if r.std(ddof=1) == 0.0 or not np.isfinite(r.std(ddof=1)):
        return
    s1 = sharpe_ratio(r, risk_free=0.0, periods_per_year=1)
    s2 = sharpe_ratio(2.0 * r, risk_free=0.0, periods_per_year=1)
    if not (np.isfinite(s1) and np.isfinite(s2)):
        return
    assert s1 == pytest.approx(s2, rel=1e-6, abs=1e-9)


@_SETTINGS
@given(r=returns_strategy)
def test_property_var_monotonic_in_confidence(r: pd.Series) -> None:
    v95 = float(value_at_risk(r, confidence=0.95, method="historical"))
    v99 = float(value_at_risk(r, confidence=0.99, method="historical"))
    # 99% confidence VaR is at least as severe (i.e., <=) as 95% VaR.
    assert v99 <= v95 + 1e-12


@_SETTINGS
@given(r=returns_strategy)
def test_property_cvar_le_var(r: pd.Series) -> None:
    v = float(value_at_risk(r, confidence=0.95, method="historical"))
    cv = float(conditional_value_at_risk(r, confidence=0.95, method="historical"))
    assert cv <= v + 1e-12


@_SETTINGS
@given(r=returns_strategy)
def test_property_volatility_nonneg(r: pd.Series) -> None:
    vol = float(realized_volatility(r, periods_per_year=252))
    assert vol >= 0.0 or math.isnan(vol)


@_SETTINGS
@given(r=returns_strategy)
def test_property_cumulative_return_floor(r: pd.Series) -> None:
    # All sampled returns are in [-0.5, 0.5], so strictly > -1, so wealth stays
    # positive and cumulative return is >= -1.
    cum = cumulative_returns(r)
    # cumulative_returns may return a Series; compare element-wise.
    arr = np.asarray(cum, dtype=float)
    assert float(arr.min()) >= -1.0 - 1e-12


@_SETTINGS
@given(r=returns_strategy)
def test_property_psr_in_unit_interval(r: pd.Series) -> None:
    p = probabilistic_sharpe_ratio(r, threshold_sr=0.0)
    val = float(p)
    if math.isnan(val):
        return
    assert 0.0 - 1e-12 <= val <= 1.0 + 1e-12
