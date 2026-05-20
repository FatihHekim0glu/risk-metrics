"""Return-level metrics: cumulative, total, CAGR, period extrema, hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import ensure_series, validate_min_obs

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "total_return",
    "cagr",
    "annualized_return_arithmetic",
    "best_period",
    "worst_period",
    "hit_rate",
    "best_year",
    "worst_year",
    "monthly_returns_table",
]


def simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple (arithmetic) period-over-period returns from a price series.

    Args:
        prices: Series of asset prices indexed by time.

    Returns:
        Series of simple returns r_t = P_t / P_{t-1} - 1, with the first
        observation dropped.

    Raises:
        TypeError: If ``prices`` is not a ``pd.Series``.
        ValueError: If ``prices`` has fewer than 2 observations.

    Example:
        >>> import pandas as pd
        >>> p = pd.Series([100, 101, 98.98, 100.46, 100.96, 99.95])
        >>> simple_returns(p).round(6).tolist()
        [0.01, -0.02, 0.015, 0.005, -0.01]
    """
    prices = ensure_series(prices)
    validate_min_obs(prices, 2)
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute continuously compounded (log) period-over-period returns.

    Args:
        prices: Series of asset prices indexed by time.

    Returns:
        Series of log returns r_t = ln(P_t / P_{t-1}), with the first
        observation dropped.

    Raises:
        TypeError: If ``prices`` is not a ``pd.Series``.
        ValueError: If ``prices`` has fewer than 2 observations.

    Example:
        >>> import numpy as np, pandas as pd
        >>> p = pd.Series([100, 101, 98.98, 100.46, 100.96, 99.95])
        >>> float(log_returns(p).iloc[0].round(6))
        0.00995
    """
    prices = ensure_series(prices)
    validate_min_obs(prices, 2)
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: ReturnsLike) -> pd.Series:
    """Compute the cumulative compounded return path.

    The result at index t is (1 + r_1)(1 + r_2)...(1 + r_t) - 1, i.e. the
    total return earned from period 1 through period t.

    Args:
        returns: Series of period returns (simple, not log).

    Returns:
        Series of cumulative returns with the same index and length as
        ``returns``.

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If any return is at or below -100% (would drive
            cumulative wealth to zero or negative).

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> cumulative_returns(r).round(6).tolist()
        [0.01, -0.0102, 0.004647, 0.009671, -0.000426]
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    mask = r <= -1
    if mask.any():
        bad_indices = r.index[mask]
        raise ValueError(
            f"Return <= -100% at indices {bad_indices.tolist()} -- "
            f"would result in non-positive wealth; clip or use log returns."
        )
    return (1 + r).cumprod() - 1


def total_return(returns: ReturnsLike) -> float:
    """Compute the total compounded return over the full series.

    Args:
        returns: Series of period returns (simple, not log).

    Returns:
        Total return as a float: (1 + r_1)(1 + r_2)...(1 + r_n) - 1.

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> round(total_return(r), 6)
        -0.000426
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    return float((1 + r).prod() - 1)


def cagr(returns: ReturnsLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compute the Compound Annual Growth Rate (geometric annualized return).

    Formula: (1 + total_return) ** (periods_per_year / n) - 1, where
    n = len(returns). This is the correct annualized growth metric -- it
    matches the wealth multiple an investor actually realizes.

    Args:
        returns: Series of period returns (simple, not log).
        periods_per_year: Number of return observations per year (e.g. 252
            for daily, 12 for monthly). Defaults to the project constant.

    Returns:
        Annualized compound growth rate as a float.

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty or if cumulative wealth
            (1 + total_return) is non-positive.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> round(cagr(r, periods_per_year=252), 6)
        -0.021227
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    n = len(r)
    tr = float((1 + r).prod() - 1)
    wealth = 1 + tr
    if wealth <= 0:
        raise ValueError("Cannot compute CAGR when cumulative wealth is non-positive.")
    return float(wealth ** (periods_per_year / n) - 1)


def annualized_return_arithmetic(
    returns: ReturnsLike, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Compute the (biased) arithmetic annualized return: mean(r) * periods_per_year.

    WARNING -- this metric is provided ONLY for teaching and side-by-side
    comparison with :func:`cagr`. It systematically OVERSTATES the
    compounded return whenever volatility is nonzero, due to Jensen's
    inequality (a.k.a. "volatility drag"): the geometric mean of a series
    of returns is always less than or equal to the arithmetic mean, with
    equality only when every return is identical. For any reporting or
    decision-making purpose, use :func:`cagr` -- it reflects the wealth
    an investor would actually have ended up with.

    Args:
        returns: Series of period returns (simple, not log).
        periods_per_year: Number of return observations per year. Defaults
            to the project constant.

    Returns:
        Arithmetic annualized return as a float.

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> round(annualized_return_arithmetic(r, periods_per_year=252), 6)
        0.0
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    return float(r.mean() * periods_per_year)


def best_period(returns: ReturnsLike) -> tuple[pd.Timestamp, float]:
    """Identify the single period with the highest return.

    Args:
        returns: Series of period returns.

    Returns:
        Tuple ``(index_label, value)`` for the maximum return. The
        ``index_label`` is whatever the underlying Series uses
        (``pd.Timestamp`` for a DatetimeIndex, otherwise the positional
        / integer label).

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> best_period(r)
        (2, 0.015)
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    idx = r.idxmax()
    return idx, float(r.loc[idx])


def worst_period(returns: ReturnsLike) -> tuple[pd.Timestamp, float]:
    """Identify the single period with the lowest return.

    Args:
        returns: Series of period returns.

    Returns:
        Tuple ``(index_label, value)`` for the minimum return. The
        ``index_label`` is whatever the underlying Series uses
        (``pd.Timestamp`` for a DatetimeIndex, otherwise the positional
        / integer label).

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> worst_period(r)
        (1, -0.02)
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    idx = r.idxmin()
    return idx, float(r.loc[idx])


def hit_rate(returns: ReturnsLike, threshold: float = 0.0) -> float:
    """Compute the fraction of periods with return strictly greater than a threshold.

    Args:
        returns: Series of period returns.
        threshold: Cutoff value. Defaults to 0.0 (fraction of positive
            periods).

    Returns:
        Fraction in [0, 1] of observations satisfying ``r > threshold``.

    Raises:
        TypeError: If ``returns`` cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> hit_rate(r)
        0.6
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    return float((r > threshold).mean())


def best_year(returns: ReturnsLike) -> tuple[int, float]:
    """Identify the calendar year with the highest compounded return.

    Resamples to year-end frequency and compounds within each year:
    ``(1 + r).resample('YE').apply(lambda x: x.prod() - 1)``.

    Args:
        returns: Series of period returns indexed by a ``DatetimeIndex``.

    Returns:
        Tuple ``(year, return)`` for the year with the highest compounded
        return.

    Raises:
        TypeError: If ``returns`` does not have a ``DatetimeIndex`` or
            cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range('2024-01-01', periods=5, freq='D')
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01], index=idx)
        >>> best_year(r)[0]
        2024
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("best_year requires a DatetimeIndex")
    yearly = (1 + r).resample("YE").apply(lambda x: x.prod() - 1)
    idx = yearly.idxmax()
    return int(idx.year), float(yearly.loc[idx])


def worst_year(returns: ReturnsLike) -> tuple[int, float]:
    """Identify the calendar year with the lowest compounded return.

    Resamples to year-end frequency and compounds within each year:
    ``(1 + r).resample('YE').apply(lambda x: x.prod() - 1)``.

    Args:
        returns: Series of period returns indexed by a ``DatetimeIndex``.

    Returns:
        Tuple ``(year, return)`` for the year with the lowest compounded
        return.

    Raises:
        TypeError: If ``returns`` does not have a ``DatetimeIndex`` or
            cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range('2024-01-01', periods=5, freq='D')
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01], index=idx)
        >>> worst_year(r)[0]
        2024
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("worst_year requires a DatetimeIndex")
    yearly = (1 + r).resample("YE").apply(lambda x: x.prod() - 1)
    idx = yearly.idxmin()
    return int(idx.year), float(yearly.loc[idx])


def monthly_returns_table(returns: ReturnsLike) -> pd.DataFrame:
    """Pivot a return series into a year by month compounded return table.

    Each row is a calendar year; columns are months 1..12 plus a trailing
    ``"YTD"`` column holding the full-year compounded return. Empty
    (year, month) cells are NaN.

    Args:
        returns: Series of period returns indexed by a ``DatetimeIndex``.

    Returns:
        DataFrame with one row per year. Columns are integers 1..12 for
        the months, followed by a string column ``"YTD"`` for the
        compounded annual return.

    Raises:
        TypeError: If ``returns`` does not have a ``DatetimeIndex`` or
            cannot be coerced to ``pd.Series``.
        ValueError: If ``returns`` is empty.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range('2024-01-01', periods=5, freq='D')
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01], index=idx)
        >>> tbl = monthly_returns_table(r)
        >>> tbl.index.tolist()
        [2024]
    """
    r = ensure_series(returns)
    validate_min_obs(r, 1)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("monthly_returns_table requires a DatetimeIndex")
    monthly = (1 + r).resample("ME").apply(lambda x: x.prod() - 1)
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }
    )
    table = frame.pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=list(range(1, 13)))
    ytd = (1 + r).groupby(r.index.year).apply(lambda x: x.prod() - 1)
    ytd.index.name = "year"
    table["YTD"] = ytd
    table.columns.name = None
    return table
