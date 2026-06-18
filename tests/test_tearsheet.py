"""Coverage for the tearsheet aggregator: row groups, benchmark rows, and the
NaN-on-failure contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.tearsheet import _safe, tearsheet


@pytest.fixture
def daily_returns() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, 400), index=idx, name="r")


@pytest.fixture
def daily_benchmark() -> pd.Series:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    return pd.Series(rng.normal(0.0004, 0.009, 400), index=idx, name="bench")


def test_tearsheet_returns_labeled_series(daily_returns: pd.Series) -> None:
    s = tearsheet(daily_returns)
    assert isinstance(s, pd.Series)
    assert s.index.name == "metric"
    assert s.name == "tearsheet"


def test_tearsheet_has_core_metric_groups(daily_returns: pd.Series) -> None:
    s = tearsheet(daily_returns)
    for key in (
        "returns.cagr",
        "volatility.annualized",
        "drawdown.max",
        "ratios.sharpe",
        "tail.var_95_historical",
    ):
        assert key in s.index
    # Benchmark rows are omitted when no benchmark is supplied.
    assert not any(k.startswith("benchmark.") for k in s.index)


def test_tearsheet_values_are_floats(daily_returns: pd.Series) -> None:
    s = tearsheet(daily_returns)
    assert s.dtype == float
    assert np.isfinite(s["returns.cagr"])
    assert s["drawdown.max"] <= 0.0


def test_tearsheet_with_benchmark_adds_benchmark_rows(
    daily_returns: pd.Series, daily_benchmark: pd.Series
) -> None:
    s = tearsheet(daily_returns, benchmark=daily_benchmark)
    for key in (
        "benchmark.beta",
        "benchmark.alpha_annualized",
        "benchmark.tracking_error",
        "benchmark.information_ratio",
        "benchmark.correlation",
        "ratios.m_squared",
    ):
        assert key in s.index
    assert np.isfinite(s["benchmark.beta"])


def test_tearsheet_handles_short_series_without_raising() -> None:
    # Too short for most metrics; the tearsheet must still return a Series of
    # NaNs rather than propagating a ValueError.
    r = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="B"))
    s = tearsheet(r)
    assert isinstance(s, pd.Series)


def test_safe_collapses_exceptions_to_nan() -> None:
    def boom() -> float:
        raise ValueError("nope")

    assert np.isnan(_safe(boom))


def test_safe_unpacks_tuple_second_element() -> None:
    assert _safe(lambda: ("2024", 0.015)) == pytest.approx(0.015, abs=1e-12)


def test_safe_returns_nan_for_non_finite_tuple_value() -> None:
    assert np.isnan(_safe(lambda: ("2024", float("nan"))))
