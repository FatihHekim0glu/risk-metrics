"""Coverage for benchmark-relative metrics beyond the hand-calc suite:
tracking error, information-ratio guards, capture edge cases, correlation,
rolling alpha, the alpha/beta wrappers, and the risk-free Series path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.benchmark import (
    alpha,
    alpha_beta,
    beta,
    correlation,
    down_capture,
    information_ratio,
    rolling_alpha,
    rolling_beta,
    tracking_error,
    up_capture,
)


@pytest.fixture
def aligned_pair() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(11)
    n = 260
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0004, 0.009, n), index=idx, name="bench")
    noise = pd.Series(rng.normal(0.0, 0.003, n), index=idx, name="noise")
    asset = (1.1 * bench + 0.0002 + noise).rename("asset")
    return asset, bench


def test_tracking_error_positive(aligned_pair: tuple[pd.Series, pd.Series]) -> None:
    a, b = aligned_pair
    te = tracking_error(a, b, periods_per_year=252)
    active = (a - b).std(ddof=1) * np.sqrt(252)
    assert te == pytest.approx(float(active), abs=1e-12)


def test_tracking_error_nan_when_active_constant() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    # Active return (asset minus benchmark) is exactly 1.0 every period, so
    # the active series has zero variance (chosen as exact floats to avoid
    # floating-point residue in the difference).
    a = pd.Series([2.0, 1.0, 3.0, 0.0, 2.0], index=idx)
    b = pd.Series([1.0, 0.0, 2.0, -1.0, 1.0], index=idx)
    assert np.isnan(tracking_error(a, b))


def test_information_ratio_nan_with_warning_on_zero_te() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    a = pd.Series([2.0, 1.0, 3.0, 0.0, 2.0], index=idx)
    b = pd.Series([1.0, 0.0, 2.0, -1.0, 1.0], index=idx)
    with pytest.warns(UserWarning, match="zero tracking error"):
        out = information_ratio(a, b)
    assert np.isnan(out)


def test_correlation_in_unit_interval(aligned_pair: tuple[pd.Series, pd.Series]) -> None:
    a, b = aligned_pair
    c = correlation(a, b)
    assert -1.0 <= c <= 1.0
    # Construction makes them strongly positively correlated.
    assert c > 0.5


def test_correlation_perfect_for_linear_relation() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    b = pd.Series(np.linspace(-0.02, 0.02, 10), index=idx)
    a = 2.0 * b  # exact linear relation
    assert correlation(a, b) == pytest.approx(1.0, abs=1e-9)


def test_up_capture_nan_when_no_up_periods() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    a = pd.Series([-0.01, -0.02, -0.03, -0.01], index=idx)
    b = pd.Series([-0.01, -0.02, -0.03, -0.01], index=idx)
    assert np.isnan(up_capture(a, b))


def test_down_capture_nan_when_no_down_periods() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    a = pd.Series([0.01, 0.02, 0.03, 0.01], index=idx)
    b = pd.Series([0.01, 0.02, 0.03, 0.01], index=idx)
    assert np.isnan(down_capture(a, b))


def test_capture_ratios_finite(aligned_pair: tuple[pd.Series, pd.Series]) -> None:
    a, b = aligned_pair
    assert np.isfinite(up_capture(a, b))
    assert np.isfinite(down_capture(a, b))


def test_beta_wrapper_matches_alpha_beta(aligned_pair: tuple[pd.Series, pd.Series]) -> None:
    a, b = aligned_pair
    assert beta(a, b) == pytest.approx(alpha_beta(a, b).beta, abs=1e-12)


def test_alpha_wrapper_annualized_and_per_period(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    per_period = alpha(a, b, annualized=False)
    annual = alpha(a, b, annualized=True)
    assert annual == pytest.approx(per_period * 252, abs=1e-9)


def test_alpha_beta_nonrobust_cov_type_runs(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    res = alpha_beta(a, b, cov_type="nonrobust")
    assert np.isfinite(res.beta)
    assert res.n_obs == len(a)


def test_alpha_beta_accepts_risk_free_series(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    rf = pd.Series(0.02, index=a.index)
    res_series = alpha_beta(a, b, risk_free=rf)
    res_scalar = alpha_beta(a, b, risk_free=0.02)
    # A flat rate cancels from both sides, so beta is unchanged.
    assert res_series.beta == pytest.approx(res_scalar.beta, abs=1e-9)


def test_alpha_beta_raises_on_non_overlapping_risk_free_series(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    # Risk-free Series indexed entirely outside the returns window.
    rf = pd.Series(0.02, index=pd.date_range("2099-01-01", periods=10, freq="B"))
    with pytest.raises(ValueError, match="no overlapping observations"):
        alpha_beta(a, b, risk_free=rf)


def test_rolling_beta_rejects_small_window(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    with pytest.raises(ValueError, match="window must be an integer"):
        rolling_beta(a, b, window=2)


def test_rolling_alpha_shape_and_annualization(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    per_period = rolling_alpha(a, b, window=60, annualized=False)
    annual = rolling_alpha(a, b, window=60, annualized=True)
    valid = per_period.dropna().index
    assert len(valid) > 0
    np.testing.assert_allclose(
        annual.loc[valid].to_numpy(),
        per_period.loc[valid].to_numpy() * 252,
        rtol=1e-9,
    )


def test_rolling_alpha_rejects_small_window(
    aligned_pair: tuple[pd.Series, pd.Series],
) -> None:
    a, b = aligned_pair
    with pytest.raises(ValueError, match="window must be an integer"):
        rolling_alpha(a, b, window=2)
