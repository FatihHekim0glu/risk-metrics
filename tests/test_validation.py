"""Input-validation and coercion edge cases for the shared helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics._validation import align_inner, ensure_series, validate_min_obs


def test_ensure_series_accepts_list() -> None:
    out = ensure_series([0.01, -0.02, 0.03])
    assert isinstance(out, pd.Series)
    assert out.dtype == float
    assert out.tolist() == [0.01, -0.02, 0.03]


def test_ensure_series_accepts_1d_ndarray() -> None:
    out = ensure_series(np.array([0.01, -0.02, 0.03]))
    assert isinstance(out, pd.Series)
    assert out.tolist() == [0.01, -0.02, 0.03]


def test_ensure_series_copies_input_series() -> None:
    original = pd.Series([0.01, 0.02], name="r")
    out = ensure_series(original)
    out.iloc[0] = 99.0
    # The copy must not mutate the caller's series.
    assert original.iloc[0] == 0.01


def test_ensure_series_rejects_2d_ndarray() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        ensure_series(np.zeros((2, 2)))


def test_ensure_series_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        ensure_series({"a": 1})  # type: ignore[arg-type]


def test_ensure_series_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        ensure_series([])


def test_ensure_series_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        ensure_series([0.01, np.nan, 0.02])


def test_ensure_series_rejects_inf() -> None:
    with pytest.raises(ValueError, match="infinite"):
        ensure_series([0.01, np.inf, 0.02])


def test_ensure_series_custom_name_in_message() -> None:
    with pytest.raises(ValueError, match="prices is empty"):
        ensure_series([], name="prices")


def test_align_inner_drops_nonoverlapping_rows() -> None:
    a = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2], name="a")
    b = pd.Series([4.0, 5.0, 6.0], index=[1, 2, 3], name="b")
    left, right = align_inner(a, b)
    assert left.index.tolist() == [1, 2]
    assert right.index.tolist() == [1, 2]
    assert left.name == "a"
    assert right.name == "b"


def test_align_inner_raises_on_no_overlap() -> None:
    a = pd.Series([1.0, 2.0], index=[0, 1])
    b = pd.Series([3.0, 4.0], index=[5, 6])
    with pytest.raises(ValueError, match="no overlapping"):
        align_inner(a, b)


def test_validate_min_obs_passes_when_enough() -> None:
    validate_min_obs(pd.Series([1.0, 2.0, 3.0]), min_obs=2)


def test_validate_min_obs_raises_with_metric_label() -> None:
    with pytest.raises(ValueError, match="sharpe requires at least 3"):
        validate_min_obs(pd.Series([1.0, 2.0]), min_obs=3, metric="sharpe")
