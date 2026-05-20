from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.benchmark import (
    alpha_beta,
    information_ratio,
    rolling_beta,
    up_capture,
)


def _make_pair(
    seed: int = 0,
    n: int = 1500,
    true_beta: float = 1.5,
    true_alpha: float = 0.0001,
    noise_sd: float = 0.003,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0005, 0.01, n), index=idx, name="bench")
    noise = pd.Series(rng.normal(0.0, noise_sd, n), index=idx, name="noise")
    asset = (true_beta * bench + true_alpha + noise).rename("asset")
    return asset, bench


def test_alpha_beta_excess_returns_convention() -> None:
    asset, bench = _make_pair(seed=42)
    result = alpha_beta(asset, bench)
    # Result is a CAPMResult dataclass; access via attributes.
    est_alpha = float(result.alpha)
    est_beta = float(result.beta)
    assert est_beta == pytest.approx(1.5, abs=0.05)
    assert est_alpha == pytest.approx(0.0001, abs=1e-4)


def test_ols_uses_add_constant() -> None:
    # If add_constant is omitted, OLS is forced through the origin and the
    # estimated alpha (the intercept) would be exactly 0. With a non-zero true
    # alpha, the estimated alpha must be non-zero.
    asset, bench = _make_pair(seed=7, true_alpha=0.001, noise_sd=0.002)
    result = alpha_beta(asset, bench)
    est_alpha = float(result.alpha)
    assert est_alpha != 0.0
    # Sanity: the estimate is positive (since true alpha is positive) and of
    # the right order of magnitude.
    assert est_alpha > 0.0


def test_information_ratio_sign_convention() -> None:
    rng = np.random.default_rng(11)
    n = 1000
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    # Add a clear positive excess return.
    asset = bench + 0.001 + pd.Series(rng.normal(0.0, 0.002, n), index=idx)
    ir = information_ratio(asset, bench)
    assert ir > 0.0


def test_rolling_beta_uses_RollingOLS_window() -> None:
    asset, bench = _make_pair(seed=3, n=400)
    window = 60
    rb = rolling_beta(asset, bench, window=window)
    assert isinstance(rb, pd.Series)
    # First window-1 observations have insufficient data -> NaN.
    assert rb.iloc[: window - 1].isna().all()
    # After the window fills, values should be finite.
    assert rb.iloc[window - 1 :].notna().all()


def test_up_capture_compounds_not_averages() -> None:
    # Construct a scenario where compounded benchmark return differs noticeably
    # from the simple arithmetic mean over up days, so the two methods give
    # different up-capture numbers.
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    bench = pd.Series(
        [0.10, 0.20, 0.10, 0.15, 0.05, -0.05, -0.10, 0.08, 0.12, 0.07],
        index=idx,
    )
    # Asset exactly tracks bench on up days -> "true" up-capture should be 1.0
    # under either definition. Now perturb so the two methods diverge.
    asset = bench.copy()
    # On the first up day, asset overshoots; on the second up day, undershoots.
    asset.iloc[0] = 0.30
    asset.iloc[1] = 0.10
    # Compute reference values under both conventions on up days only.
    up_mask = bench > 0
    a_up = asset[up_mask]
    b_up = bench[up_mask]

    compounded = ((1.0 + a_up).prod() - 1.0) / ((1.0 + b_up).prod() - 1.0)
    averaged = a_up.mean() / b_up.mean()
    # Sanity: our construction yields two clearly distinct candidates.
    assert not np.isclose(compounded, averaged, atol=1e-6)

    got = up_capture(asset, bench)
    # The function must match the compounding convention.
    assert got == pytest.approx(compounded, rel=1e-6, abs=1e-9)
