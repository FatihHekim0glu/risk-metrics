"""Volatility and dispersion metrics: realized vol, downside deviation, semi-deviation, MAD, rolling vol, autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import ensure_series, validate_min_obs

__all__ = [
    "realized_volatility",
    "downside_deviation",
    "semi_deviation",
    "mean_absolute_deviation",
    "rolling_volatility",
    "autocorrelation",
    "ljung_box_test",
]


def realized_volatility(
    returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualize: bool = True,
    ddof: int = 1,
) -> float:
    """Realized (sample) volatility of a return series.

    Computes the sample standard deviation of returns, optionally annualized
    by the square-root-of-time rule. Annualization for std-based metrics uses
    sqrt(periods_per_year) under the iid assumption.

    Args:
        returns: Periodic return series (Series, ndarray, or sequence).
        periods_per_year: Number of periods per year used for annualization
            (e.g. 252 for daily, 12 for monthly).
        annualize: If True, multiply the per-period std by sqrt(periods_per_year).
        ddof: Delta degrees of freedom passed to ``Series.std``. Default 1
            corresponds to the unbiased sample standard deviation.

    Returns:
        Realized volatility as a float. NaN if the input has fewer than 2
        non-NaN observations.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations after validation.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> realized_volatility(r, periods_per_year=252)  # doctest: +ELLIPSIS
        0.2...
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2)
    sigma = float(r.std(ddof=ddof))
    if annualize:
        sigma *= float(np.sqrt(periods_per_year))
    return sigma


def downside_deviation(
    returns: ReturnsLike,
    mar: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualize: bool = True,
) -> float:
    """Downside deviation against a minimum acceptable return (MAR).

    Follows Sortino & van der Meer (1991) / Sortino & Price (1994) convention:
    divisor is total N, not number of below-MAR observations. This is the
    convention used by empyrical, quantstats, pyfolio, and vectorbt.

    Concretely, with shortfall ``s_t = min(r_t - mar, 0)``:

        dd = sqrt( sum(s_t**2) / N )

    Annualization uses sqrt(periods_per_year) since downside deviation is a
    standard-deviation-like quantity that scales with sqrt(T) under iid.

    Args:
        returns: Periodic return series.
        mar: Minimum acceptable return per period (same units as ``returns``).
            Defaults to 0.
        periods_per_year: Number of periods per year used for annualization.
        annualize: If True, multiply the per-period downside deviation by
            sqrt(periods_per_year).

    Returns:
        Downside deviation as a float. Zero if no observation falls below MAR.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> downside_deviation(r, mar=0.0, periods_per_year=252)  # doctest: +ELLIPSIS
        0.1...
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2)
    shortfall = (r - mar).clip(upper=0.0)
    dd = float(np.sqrt((shortfall**2).sum() / len(r)))
    if annualize:
        dd *= float(np.sqrt(periods_per_year))
    return dd


def semi_deviation(
    returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualize: bool = True,
) -> float:
    """Semi-deviation: downside deviation with MAR set to the sample mean.

    Identical to :func:`downside_deviation` but with ``mar = mean(returns)``.
    Useful for measuring dispersion below the average return.

    Divisor is total N (not number of below-mean observations), matching the
    Sortino convention used elsewhere in this module. Annualization uses
    sqrt(periods_per_year).

    Args:
        returns: Periodic return series.
        periods_per_year: Number of periods per year used for annualization.
        annualize: If True, multiply by sqrt(periods_per_year).

    Returns:
        Semi-deviation as a float.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> semi_deviation(r, periods_per_year=252)  # doctest: +ELLIPSIS
        0.1...
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2)
    mar = float(r.mean())
    return downside_deviation(
        r,
        mar=mar,
        periods_per_year=periods_per_year,
        annualize=annualize,
    )


def mean_absolute_deviation(
    returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualize: bool = True,
) -> float:
    """Mean absolute deviation (MAD) of returns from their mean.

    Computes ``mean(|r - mean(r)|)``. Unlike standard deviation, MAD scales
    linearly with time under iid (not with sqrt(T)), so annualization
    multiplies by ``periods_per_year`` (linear), NOT ``sqrt(periods_per_year)``.

    Args:
        returns: Periodic return series.
        periods_per_year: Number of periods per year used for annualization.
        annualize: If True, multiply the per-period MAD by ``periods_per_year``
            (linear scaling).

    Returns:
        Mean absolute deviation as a float.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> mean_absolute_deviation(r, periods_per_year=252)  # doctest: +ELLIPSIS
        2...
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2)
    mad = float((r - r.mean()).abs().mean())
    if annualize:
        mad *= float(periods_per_year)
    return mad


def rolling_volatility(
    returns: ReturnsLike,
    window: int = PERIODS_PER_YEAR,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualize: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling (annualized) realized volatility.

    Computes a rolling sample standard deviation over a fixed window and
    optionally annualizes by sqrt(periods_per_year). Annualization for
    std-based metrics uses sqrt(T) under the iid assumption.

    NaN handling: if ``min_periods is None`` it defaults to ``window`` so the
    series is NaN until the window is fully populated and every reported
    value is computed from a consistent window size. Defaulting to
    ``min_periods=1`` would produce inconsistent-window series and is
    intentionally not used. Windows containing any NaN return NaN
    (enforced via ``raw=True`` with a NaN-rejecting lambda).

    Args:
        returns: Periodic return series.
        window: Rolling window size in periods.
        periods_per_year: Number of periods per year used for annualization.
        annualize: If True, multiply each rolling std by sqrt(periods_per_year).
        min_periods: Minimum observations in a window required to emit a
            value. If None, defaults to ``window`` (fully populated windows
            only).

    Returns:
        ``pd.Series`` of rolling (optionally annualized) volatility, indexed
        like the input.

    Raises:
        ValueError: If the input cannot be coerced to a Series, ``window`` is
            not a positive integer, or there are fewer observations than the
            window.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> rolling_volatility(r, window=3, periods_per_year=252).dropna().iloc[0]  # doctest: +ELLIPSIS
        0.2...
    """
    r = ensure_series(returns)
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    validate_min_obs(r, min_obs=window)
    mp = window if min_periods is None else int(min_periods)

    def _std(w: np.ndarray) -> float:
        if np.isnan(w).any():
            return float("nan")
        return float(np.std(w, ddof=1))

    rolled = r.rolling(window=window, min_periods=mp).apply(_std, raw=True)
    if annualize:
        rolled = rolled * float(np.sqrt(periods_per_year))
    return rolled


def autocorrelation(returns: ReturnsLike, lag: int = 1) -> float:
    """Sample autocorrelation of returns at a given lag.

    Thin wrapper around ``pd.Series.autocorr``. Useful for diagnosing
    serial dependence; if returns are autocorrelated, sqrt(T) annualization
    of volatility understates (positive autocorr) or overstates (negative
    autocorr) the true annualized risk.

    Args:
        returns: Periodic return series.
        lag: Lag at which to compute the autocorrelation. Default 1.

    Returns:
        Autocorrelation coefficient at the given lag, as a float in [-1, 1].
        NaN if the input is too short for the requested lag.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than ``lag + 2`` observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> autocorrelation(r, lag=1)  # doctest: +ELLIPSIS
        -0...
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=lag + 2)
    return float(r.autocorr(lag=lag))


def ljung_box_test(returns: ReturnsLike, lags: int = 5) -> tuple[float, float]:
    """Ljung-Box portmanteau test for serial correlation.

    Tests the null hypothesis that the first ``lags`` autocorrelations of
    the return series are jointly zero. A low p-value indicates the
    presence of autocorrelation, which is relevant to whether the sqrt(T)
    scaling rule for volatility is valid.

    Args:
        returns: Periodic return series.
        lags: Number of lags to include in the joint test. Default 5.

    Returns:
        Tuple ``(stat, p_value)`` where ``stat`` is the Ljung-Box Q statistic
        and ``p_value`` is the associated chi-squared p-value.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than ``lags + 2`` observations.
        ImportError: If ``statsmodels`` is not installed.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> stat, p = ljung_box_test(r, lags=2)
        >>> isinstance(stat, float) and isinstance(p, float)
        True
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = ensure_series(returns)
    validate_min_obs(r, min_obs=lags + 2)
    result = acorr_ljungbox(r, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])
    return stat, p_value
