from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from riskmetrics.tail import (
    conditional_value_at_risk,
    cornish_fisher_var,
    excess_kurtosis,
    jarque_bera,
    probabilistic_sharpe_ratio,
    value_at_risk,
)


def test_var_historical_uses_lower_method(tiny_returns: pd.Series) -> None:
    # Historical VaR at 80% on a 5-obs sample should pick one of the realized
    # returns under the "lower" interpolation convention (no interpolation).
    val = value_at_risk(tiny_returns, confidence=0.80, method="historical")
    observed = set(tiny_returns.tolist())
    assert any(abs(val - o) < 1e-12 for o in observed)


def test_var_negative_sign() -> None:
    rng = np.random.default_rng(5)
    n = 1500
    series = pd.Series(rng.normal(0.0005, 0.012, n))
    v = value_at_risk(series, confidence=0.95, method="historical")
    assert v < 0.0


def test_cornish_fisher_warns_outside_domain() -> None:
    # Construct a heavily negatively-skewed series (skew < -1).
    rng = np.random.default_rng(123)
    base = rng.normal(0.0, 0.005, 1500)
    # Inject large negative shocks to push skew below -1.
    shocks_idx = rng.choice(1500, size=40, replace=False)
    base[shocks_idx] -= 0.06
    series = pd.Series(base)
    # Sanity precondition: skewness should indeed be negative and < -1.
    # (We don't assert it strictly to avoid coupling to a particular skew impl,
    # but the construction reliably yields skew < -1.)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = cornish_fisher_var(series, confidence=0.99)
        assert any(issubclass(rec.category, UserWarning) for rec in w)


def test_cvar_le_var() -> None:
    rng = np.random.default_rng(8)
    n = 600
    series = pd.Series(rng.normal(0.0, 0.01, n))
    v = value_at_risk(series, confidence=0.95, method="historical")
    cv = conditional_value_at_risk(series, confidence=0.95, method="historical")
    # CVaR is the average loss in the tail beyond VaR, so it's at most VaR
    # (i.e., more negative or equal).
    assert cv <= v + 1e-12


def test_psr_in_unit_interval() -> None:
    rng = np.random.default_rng(13)
    series = pd.Series(rng.normal(0.0005, 0.01, 1000))
    p = probabilistic_sharpe_ratio(series, threshold_sr=0.0)
    assert 0.0 <= p <= 1.0


def test_jarque_bera_normal_data_high_p() -> None:
    rng = np.random.default_rng(2024)
    series = pd.Series(rng.normal(0.0, 1.0, 5000))
    result = jarque_bera(series)
    # Accept either a (stat, p) tuple or a result object with a .pvalue attribute.
    if hasattr(result, "pvalue"):
        p_value = float(result.pvalue)
    else:
        _, p_value = result
        p_value = float(p_value)
    assert p_value > 0.05


def test_excess_kurtosis_normal_near_zero() -> None:
    rng = np.random.default_rng(7)
    series = pd.Series(rng.normal(0.0, 1.0, 5000))
    ek = excess_kurtosis(series)
    assert abs(ek) < 0.2
