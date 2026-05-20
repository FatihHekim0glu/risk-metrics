"""Input-validation helpers shared by the public metric functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._typing import ReturnsLike


def ensure_series(x: ReturnsLike, name: str = "returns") -> pd.Series:
    """Coerce ``x`` to a :class:`pandas.Series` and check it is finite and non-empty.

    Lists and NumPy arrays are converted to a Series with a default integer
    index; an existing Series is returned (a shallow copy) unchanged. Empty
    inputs and any NaN or infinite values raise :class:`ValueError`.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional, got ndim={x.ndim}")
        s = pd.Series(x)
    elif isinstance(x, list):
        s = pd.Series(x, dtype=float)
    else:
        raise TypeError(
            f"{name} must be a pandas Series, numpy ndarray, or list of floats; "
            f"got {type(x).__name__}"
        )

    if len(s) == 0:
        raise ValueError(f"{name} is empty")

    values = s.to_numpy()
    if np.any(pd.isna(values)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} contains infinite values")

    return s


def align_inner(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-align two Series on their index and drop rows with NaN in either.

    Raises :class:`ValueError` if the intersection is empty after dropping
    rows with missing values on either side.
    """
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError("aligned series have no overlapping non-NaN observations")
    left = joined.iloc[:, 0]
    right = joined.iloc[:, 1]
    left.name = a.name
    right.name = b.name
    return left, right


def validate_min_obs(s: pd.Series, min_obs: int, metric: str) -> None:
    """Raise :class:`ValueError` if ``s`` has fewer than ``min_obs`` observations."""
    if len(s) < min_obs:
        raise ValueError(
            f"{metric} requires at least {min_obs} observations, got {len(s)}"
        )
