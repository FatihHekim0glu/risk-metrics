"""Risk-adjusted return ratios: Sharpe (with optional Lo-2002 autocorrelation adjustment), Sortino (N-divisor convention), Omega, Treynor, M²."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import ensure_series, validate_min_obs


def _to_per_period_rf(
    risk_free: float | pd.Series,
    returns: pd.Series,
    periods_per_year: int,
) -> pd.Series:
    """Convert an annual risk-free input to a per-period Series aligned to ``returns.index``.

    A scalar ``risk_free`` is interpreted as an annual rate and broadcast across
    the returns index. A Series input is inner-aligned to the returns index,
    forward-filled to cover any internal gaps, then converted from annual to
    per-period using the geometric relation ``(1 + rf)**(1/q) - 1`` where
    ``q = periods_per_year``.
    """
    if isinstance(risk_free, pd.Series):
        aligned = risk_free.reindex(returns.index).ffill()
        aligned = aligned.dropna()
        if aligned.empty:
            raise ValueError("risk_free Series has no overlapping observations with returns")
        # Re-align returns to the surviving rf index after dropping leading NaNs.
        per_period = (1.0 + aligned) ** (1.0 / periods_per_year) - 1.0
        return per_period.reindex(returns.index).ffill().bfill()

    rf_pp = (1.0 + float(risk_free)) ** (1.0 / periods_per_year) - 1.0
    return pd.Series(rf_pp, index=returns.index, dtype=float)


def sharpe_ratio(
    returns: ReturnsLike,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    smart: bool = False,
) -> float:
    """Annualised Sharpe ratio, with an optional Lo (2002) autocorrelation adjustment.

    With ``smart=False`` (default), the ratio is the mean of the per-period
    excess return divided by its sample standard deviation (``ddof=1``), then
    scaled by ``sqrt(periods_per_year)``. With ``smart=True``, the naive
    annualisation factor ``sqrt(q)`` is replaced by the Lo (2002) factor

        ``η(q) = q / sqrt(q + 2 * Σ_{k=1}^{q-1} (q - k) * ρ_k)``

    where ``ρ_k`` is the lag-``k`` autocorrelation of the excess-return series.
    For computational efficiency the inner sum is truncated at
    ``min(q - 1, len(returns) - 1)``. The smart variant corrects for the
    Madoff-style inflation of Sharpe under positively autocorrelated returns.

    Args:
        returns: Periodic returns (e.g. daily). 1-D Series, ndarray, or list.
        risk_free: Annual risk-free rate as a float, or an annual-rate Series
            indexed like ``returns`` (will be aligned and forward-filled).
            Converted to per-period geometrically.
        periods_per_year: Annualisation factor (252 for daily, 12 for monthly).
        smart: If True, apply the Lo (2002) autocorrelation adjustment.

    Returns:
        The annualised Sharpe ratio as a float. Returns ``np.nan`` if the
        excess-return series has zero volatility.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        >>> round(sharpe_ratio(r, risk_free=0.0, periods_per_year=252), 4)
        6.7082
    """
    r = ensure_series(returns, name="returns")
    validate_min_obs(r, min_obs=2, metric="sharpe_ratio")

    rf_pp = _to_per_period_rf(risk_free, r, periods_per_year)
    excess = r - rf_pp

    std = excess.std(ddof=1)
    # Use a small tolerance for the "zero volatility" branch -- subtracting
    # a constant risk-free can introduce ~1e-19 floating-point noise even when
    # the underlying series is a constant. The threshold is chosen well below
    # any meaningful daily volatility (typical equity sigma ~ 1e-2).
    if not np.isfinite(std) or std < 1e-15:
        warnings.warn("zero volatility; Sharpe undefined", UserWarning, stacklevel=2)
        return float("nan")

    mean_excess = excess.mean()

    if not smart:
        return float(mean_excess / std * np.sqrt(periods_per_year))

    q = int(periods_per_year)
    n = len(excess)
    max_lag = min(q - 1, n - 1)
    excess_arr = excess.to_numpy(dtype=float)
    mu = excess_arr.mean()
    centred = excess_arr - mu
    denom_var = float((centred * centred).sum())

    rho_sum = 0.0
    if denom_var > 0.0:
        for k in range(1, max_lag + 1):
            cov_k = float((centred[k:] * centred[:-k]).sum())
            rho_k = cov_k / denom_var
            rho_sum += (q - k) * rho_k

    denom_inside = q + 2.0 * rho_sum
    if denom_inside <= 0.0 or not np.isfinite(denom_inside):
        warnings.warn(
            "Lo (2002) adjustment denominator non-positive; returning NaN",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    eta = q / np.sqrt(denom_inside)
    return float(mean_excess / std * eta)


def sortino_ratio(
    returns: ReturnsLike,
    risk_free: float | pd.Series = 0.0,
    mar: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio.

    Divisor is total N, not N_downside (Sortino & Price 1994 convention). The
    numerator is the annualised mean excess return over the per-period
    risk-free rate; the denominator is the annualised downside deviation
    measured against ``mar``.

    Args:
        returns: Periodic returns. 1-D Series, ndarray, or list.
        risk_free: Annual risk-free rate (float or annual-rate Series).
            Converted to per-period geometrically.
        mar: Minimum acceptable return threshold for the downside deviation,
            expressed per period. Defaults to 0.
        periods_per_year: Annualisation factor.

    Returns:
        The annualised Sortino ratio as a float. Returns ``np.nan`` if there
        is no downside deviation.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        >>> round(sortino_ratio(r, risk_free=0.0, periods_per_year=252), 4)
        17.3897
    """
    r = ensure_series(returns, name="returns")
    validate_min_obs(r, min_obs=2, metric="sortino_ratio")

    rf_pp = _to_per_period_rf(risk_free, r, periods_per_year)
    excess = r - rf_pp

    shortfall = (r - mar).clip(upper=0.0)
    dd = float(np.sqrt((shortfall**2).sum() / len(r)) * np.sqrt(periods_per_year))

    if dd == 0.0 or not np.isfinite(dd):
        warnings.warn("zero downside deviation; Sortino undefined", UserWarning, stacklevel=2)
        return float("nan")

    numerator = float(excess.mean() * periods_per_year)
    return numerator / dd


def omega_ratio(returns: ReturnsLike, threshold: float = 0.0) -> float:
    """Omega ratio against a return threshold.

    Computed as ``sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))``.
    Returns ``+inf`` when the denominator is zero.

    Args:
        returns: Periodic returns. 1-D Series, ndarray, or list.
        threshold: Per-period return threshold separating gains from losses.

    Returns:
        The Omega ratio as a float. Returns ``+np.inf`` if no observations
        fall below the threshold.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        >>> round(omega_ratio(r, threshold=0.0), 4)
        3.0
    """
    r = ensure_series(returns, name="returns")

    gains = (r - threshold).clip(lower=0.0).sum()
    losses = (threshold - r).clip(lower=0.0).sum()

    if losses == 0.0:
        return float("inf")
    return float(gains / losses)


def treynor_ratio(
    returns: ReturnsLike,
    beta: float,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Treynor ratio: annualised excess return divided by beta.

    Beta is supplied by the caller (typically from ``benchmark.alpha_beta``)
    to avoid a circular dependency on the benchmark module.

    Args:
        returns: Periodic returns. 1-D Series, ndarray, or list.
        beta: Portfolio beta against the relevant benchmark.
        risk_free: Annual risk-free rate (float or annual-rate Series).
            Converted to per-period geometrically.
        periods_per_year: Annualisation factor.

    Returns:
        The Treynor ratio as a float. Returns ``np.nan`` with a
        ``UserWarning`` if ``beta`` is zero or NaN.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        >>> round(treynor_ratio(r, beta=1.0, periods_per_year=252), 4)
        1.26
    """
    r = ensure_series(returns, name="returns")
    validate_min_obs(r, min_obs=1, metric="treynor_ratio")

    if beta is None or not np.isfinite(beta) or beta == 0.0:
        warnings.warn("beta is zero or NaN; Treynor undefined", UserWarning, stacklevel=2)
        return float("nan")

    rf_pp = _to_per_period_rf(risk_free, r, periods_per_year)
    excess = r - rf_pp
    annualised_excess = float(excess.mean() * periods_per_year)
    return annualised_excess / float(beta)


def m_squared(
    returns: ReturnsLike,
    benchmark_vol_annualized: float,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Modigliani M² measure.

    Computed as ``M² = SR * sigma_benchmark + rf_annual`` where ``SR`` is the
    plain (non-smart) annualised Sharpe ratio and ``rf_annual`` is the sample
    average annual risk-free rate (the input itself if scalar, otherwise the
    mean of the aligned Series).

    Args:
        returns: Periodic returns. 1-D Series, ndarray, or list.
        benchmark_vol_annualized: Annualised volatility of the benchmark
            (standard deviation of benchmark returns scaled by
            ``sqrt(periods_per_year)``).
        risk_free: Annual risk-free rate (float or annual-rate Series).
            A Series is averaged over the sample to a single annual rate.
        periods_per_year: Annualisation factor.

    Returns:
        The M² value as a float, expressed as an annualised return.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        >>> round(m_squared(r, benchmark_vol_annualized=0.16), 4)
        1.0733
    """
    r = ensure_series(returns, name="returns")

    sr = sharpe_ratio(
        r,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        smart=False,
    )

    if isinstance(risk_free, pd.Series):
        aligned = risk_free.reindex(r.index).ffill().dropna()
        rf_annual_avg = float(aligned.mean()) if not aligned.empty else 0.0
    else:
        rf_annual_avg = float(risk_free)

    return float(sr * float(benchmark_vol_annualized) + rf_annual_avg)
