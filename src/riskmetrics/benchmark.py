"""Benchmark-relative metrics: alpha, beta (excess-on-excess CAPM with HAC errors), R², tracking error, information ratio, up/down capture, rolling versions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import align_inner, ensure_series, validate_min_obs

__all__ = [
    "CAPMResult",
    "alpha_beta",
    "beta",
    "alpha",
    "tracking_error",
    "information_ratio",
    "up_capture",
    "down_capture",
    "rolling_beta",
    "rolling_alpha",
    "correlation",
]


@dataclass(frozen=True)
class CAPMResult:
    """Result of an excess-on-excess CAPM regression.

    Attributes:
        alpha: Per-period intercept (e.g. daily alpha).
        beta: Slope coefficient on the benchmark excess return.
        alpha_annualized: ``alpha * periods_per_year`` (linear annualisation of
            the per-period mean, matching the empyrical/quantstats convention).
        alpha_tstat: t-statistic on the intercept under the chosen covariance.
        alpha_pvalue: p-value associated with ``alpha_tstat``.
        r_squared: Coefficient of determination of the fitted regression.
        n_obs: Number of observations used after inner-alignment.
    """

    alpha: float
    beta: float
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    n_obs: int


def _newey_west_lags(n: int) -> int:
    """Default Newey-West lag selector ``floor(4 * (n/100) ** (2/9))``.

    This is the standard plug-in rule from Newey & West (1994) used by most
    econometrics packages when the user does not specify a bandwidth.
    """
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _to_per_period_rf(
    risk_free: float | pd.Series,
    index: pd.Index,
    periods_per_year: int,
) -> pd.Series:
    """Convert an annual risk-free input to a per-period Series aligned to ``index``.

    A scalar ``risk_free`` is interpreted as an annual rate and broadcast
    across ``index``. A Series input is reindexed to ``index``, forward-filled
    to cover internal gaps, and then converted from annual to per-period using
    the geometric relation ``(1 + rf) ** (1 / q) - 1``.
    """
    if isinstance(risk_free, pd.Series):
        aligned = risk_free.reindex(index).ffill()
        if aligned.dropna().empty:
            raise ValueError("risk_free Series has no overlapping observations with returns")
        per_period = (1.0 + aligned) ** (1.0 / periods_per_year) - 1.0
        return per_period.ffill().bfill()

    rf_pp = (1.0 + float(risk_free)) ** (1.0 / periods_per_year) - 1.0
    return pd.Series(rf_pp, index=index, dtype=float)


def alpha_beta(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    cov_type: str = "HAC",
    hac_lags: int | None = None,
) -> CAPMResult:
    """Excess-on-excess CAPM regression of asset on benchmark.

    Fits ``r_a - rf = alpha + beta * (r_b - rf) + e`` by ordinary least
    squares with an explicit constant. Standard errors default to
    Newey-West (HAC) with the Newey-West (1994) plug-in lag selector
    ``floor(4 * (n / 100) ** (2/9))``, which is robust to the mild serial
    correlation and heteroskedasticity typical of daily financial returns.

    Args:
        asset_returns: Periodic asset returns. 1-D Series, ndarray, or list.
        benchmark_returns: Periodic benchmark returns; inner-aligned to
            ``asset_returns`` on their common index.
        risk_free: Annual risk-free rate as a float, or an annual-rate Series
            indexed like the asset; converted to per-period geometrically and
            forward-filled across gaps.
        periods_per_year: Annualisation factor (252 for daily, 12 for
            monthly).
        cov_type: ``statsmodels`` covariance type. ``"HAC"`` (default) yields
            Newey-West standard errors; pass ``"nonrobust"`` for classical OLS
            or ``"HC0"``/``"HC3"`` for White errors.
        hac_lags: Lag truncation for HAC errors. If ``None``, the Newey-West
            plug-in default is used. Ignored when ``cov_type`` is not
            ``"HAC"``.

    Returns:
        A :class:`CAPMResult` containing per-period alpha, beta, annualised
        alpha, the intercept t-statistic and p-value, R², and the number of
        observations used.

    Raises:
        ValueError: If either input is empty, contains NaN/inf, or the two
            series have fewer than 3 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008, 0.018])
        >>> res = alpha_beta(a, b, periods_per_year=252)
        >>> isinstance(res.beta, float)
        True
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)
    validate_min_obs(a_aligned, min_obs=3, metric="alpha_beta")

    n = len(a_aligned)
    rf_pp = _to_per_period_rf(risk_free, a_aligned.index, periods_per_year)

    y = (a_aligned - rf_pp).to_numpy(dtype=float)
    x = (b_aligned - rf_pp).to_numpy(dtype=float)

    # Always include an explicit constant: omitting it would force alpha = 0
    # silently and bias beta. This is the central correctness guard of the
    # whole module.
    x_with_const = sm.add_constant(x, has_constant="add")

    cov_kwds: dict[str, int] | None = None
    if cov_type == "HAC":
        lags = _newey_west_lags(n) if hac_lags is None else int(hac_lags)
        cov_kwds = {"maxlags": max(lags, 0)}

    model = sm.OLS(y, x_with_const)
    if cov_kwds is not None:
        res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    else:
        res = model.fit(cov_type=cov_type)

    # statsmodels labels coefficients ``const`` and ``x1`` for ndarray inputs
    # after ``add_constant``; access positionally to stay robust.
    alpha_pp = float(res.params[0])
    beta_val = float(res.params[1])
    t_alpha = float(res.tvalues[0])
    p_alpha = float(res.pvalues[0])
    r2 = float(res.rsquared)

    return CAPMResult(
        alpha=alpha_pp,
        beta=beta_val,
        alpha_annualized=alpha_pp * periods_per_year,
        alpha_tstat=t_alpha,
        alpha_pvalue=p_alpha,
        r_squared=r2,
        n_obs=n,
    )


def beta(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    risk_free: float | pd.Series = 0.0,
) -> float:
    """CAPM beta of asset versus benchmark (convenience wrapper).

    Equivalent to ``alpha_beta(...).beta``; uses HAC standard errors under
    the hood (although only the point estimate is returned here, so the
    standard-error choice does not affect the result).

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        risk_free: Annual risk-free rate, float or Series.

    Returns:
        The CAPM beta as a float.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have fewer
            than 3 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008, 0.018])
        >>> isinstance(beta(a, b), float)
        True
    """
    return alpha_beta(
        asset_returns,
        benchmark_returns,
        risk_free=risk_free,
    ).beta


def alpha(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualized: bool = True,
) -> float:
    """CAPM alpha of asset versus benchmark (convenience wrapper).

    Returns the intercept of the excess-on-excess regression, either per
    period or linearly annualised.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        risk_free: Annual risk-free rate, float or Series.
        periods_per_year: Annualisation factor used for the risk-free
            conversion and (if requested) for scaling alpha.
        annualized: If True (default), return ``alpha * periods_per_year``;
            otherwise return the per-period intercept.

    Returns:
        Alpha as a float.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have fewer
            than 3 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008, 0.018])
        >>> isinstance(alpha(a, b, annualized=True), float)
        True
    """
    res = alpha_beta(
        asset_returns,
        benchmark_returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
    )
    return res.alpha_annualized if annualized else res.alpha


def tracking_error(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised tracking error (GIPS convention).

    Defined as the sample standard deviation (``ddof=1``) of the simple
    active return ``asset - benchmark``, scaled by ``sqrt(periods_per_year)``.
    This is the GIPS / CFA Institute definition; it differs from the
    residual standard error of a CAPM regression, which is not used here.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised tracking error as a float. ``np.nan`` if the active
        series has zero variance.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have fewer
            than 2 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008])
        >>> tracking_error(a, b, periods_per_year=252) > 0
        True
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)
    validate_min_obs(a_aligned, min_obs=2, metric="tracking_error")

    active = a_aligned - b_aligned
    std = float(active.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return std * float(np.sqrt(periods_per_year))


def information_ratio(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised information ratio (GIPS convention).

    Computed as ``(mean(active) * periods_per_year) / tracking_error`` where
    ``active = asset - benchmark``. The sign convention is asset minus
    benchmark (positive when the asset outperforms); do not flip.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised information ratio as a float. ``np.nan`` with a
        ``UserWarning`` if the tracking error is zero.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have fewer
            than 2 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008])
        >>> isinstance(information_ratio(a, b), float)
        True
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)
    validate_min_obs(a_aligned, min_obs=2, metric="information_ratio")

    active = a_aligned - b_aligned
    te = float(active.std(ddof=1))
    if not np.isfinite(te) or te == 0.0:
        warnings.warn(
            "zero tracking error; information ratio undefined",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(active.mean() * periods_per_year) / (te * float(np.sqrt(periods_per_year)))


def _capture(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    periods_per_year: int,
    mask_fn,
) -> float:
    """Shared compounding capture-ratio core for up/down capture.

    Compounds both sides over the masked periods, annualises each by raising
    to ``periods_per_year / k`` (where ``k`` is the number of selected
    observations), and returns ``asset_ann / benchmark_ann``. Returns
    ``np.nan`` when the mask selects no observations or when the benchmark
    side annualises to zero.
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)

    mask = mask_fn(b_aligned)
    if not mask.any():
        return float("nan")

    a_sel = a_aligned[mask]
    b_sel = b_aligned[mask]
    k = int(mask.sum())

    asset_cum = float((1.0 + a_sel).prod())
    bench_cum = float((1.0 + b_sel).prod())

    asset_ann = asset_cum ** (periods_per_year / k) - 1.0
    bench_ann = bench_cum ** (periods_per_year / k) - 1.0

    if bench_ann == 0.0 or not np.isfinite(bench_ann):
        return float("nan")
    return asset_ann / bench_ann


def up_capture(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Up-market capture ratio.

    On periods where ``benchmark > 0``, compound the asset and benchmark
    return streams separately, annualise each, and return
    ``asset_annualised / benchmark_annualised``. A value above 1 indicates
    the asset captures more than 100% of the benchmark's up moves.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        periods_per_year: Annualisation factor used when annualising the
            compounded returns of the selected sub-period.

    Returns:
        Up-capture ratio as a float, or ``np.nan`` if there are no
        up-market periods or the annualised benchmark return is zero.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have no
            overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008])
        >>> isinstance(up_capture(a, b), float)
        True
    """
    return _capture(
        asset_returns,
        benchmark_returns,
        periods_per_year,
        mask_fn=lambda s: s > 0,
    )


def down_capture(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Down-market capture ratio.

    On periods where ``benchmark < 0``, compound the asset and benchmark
    return streams separately, annualise each, and return
    ``asset_annualised / benchmark_annualised``. A value below 1 indicates
    the asset falls less than the benchmark in down markets (good); above
    1 indicates the asset falls more (bad). Note the ratio is signed: both
    numerator and denominator are typically negative, so a positive value
    less than 1 represents protection on the downside.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        periods_per_year: Annualisation factor used when annualising the
            compounded returns of the selected sub-period.

    Returns:
        Down-capture ratio as a float, or ``np.nan`` if there are no
        down-market periods or the annualised benchmark return is zero.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have no
            overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008])
        >>> isinstance(down_capture(a, b), float)
        True
    """
    return _capture(
        asset_returns,
        benchmark_returns,
        periods_per_year,
        mask_fn=lambda s: s < 0,
    )


def rolling_beta(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    window: int = PERIODS_PER_YEAR,
    risk_free: float | pd.Series = 0.0,
) -> pd.Series:
    """Rolling CAPM beta via :class:`statsmodels.regression.rolling.RollingOLS`.

    Each window fits an excess-on-excess OLS with an explicit constant and
    extracts the slope coefficient. Using :class:`RollingOLS` is O(T) rather
    than the O(T · W) cost of ``df.rolling().apply`` with a per-window
    regression.

    Values are as-of T (inclusive); for tradable signals call ``.shift(1)``
    on the returned series so a beta computed on data up to and including
    day T is not used to trade on day T.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        window: Rolling window size in periods.
        risk_free: Annual risk-free rate, float or Series.

    Returns:
        ``pd.Series`` of beta estimates indexed by the aligned index;
        positions before the window is fully populated are NaN.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, the window is not
            a positive integer at least 3, or there are fewer observations
            than the window.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, -0.005, 0.01, 0.003], index=idx)
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008, 0.018, 0.001, -0.004, 0.009, 0.002], index=idx)
        >>> rolling_beta(a, b, window=5).dropna().shape[0]
        6
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)

    if not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3")
    validate_min_obs(a_aligned, min_obs=window, metric="rolling_beta")

    rf_pp = _to_per_period_rf(risk_free, a_aligned.index, PERIODS_PER_YEAR)
    y = (a_aligned - rf_pp).astype(float)
    x = (b_aligned - rf_pp).astype(float)
    x_with_const = sm.add_constant(x.to_frame(name="bench"), has_constant="add")

    model = RollingOLS(endog=y.to_numpy(), exog=x_with_const.to_numpy(), window=window)
    res = model.fit()
    # Column 0 is the constant, column 1 is the benchmark slope.
    beta_series = pd.Series(res.params[:, 1], index=a_aligned.index, name="beta")
    return beta_series


def rolling_alpha(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    window: int = PERIODS_PER_YEAR,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    annualized: bool = True,
) -> pd.Series:
    """Rolling CAPM alpha via :class:`statsmodels.regression.rolling.RollingOLS`.

    Each window fits the same excess-on-excess regression as
    :func:`alpha_beta` and extracts the intercept. If ``annualized`` is
    True (default) the per-period intercepts are scaled by
    ``periods_per_year``.

    Values are as-of T (inclusive); for tradable signals call ``.shift(1)``
    on the returned series.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.
        window: Rolling window size in periods.
        risk_free: Annual risk-free rate, float or Series.
        periods_per_year: Annualisation factor for the risk-free conversion
            and for scaling the alpha series when ``annualized`` is True.
        annualized: If True (default), multiply each intercept by
            ``periods_per_year``.

    Returns:
        ``pd.Series`` of alpha estimates indexed by the aligned index;
        positions before the window is fully populated are NaN.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, the window is not
            a positive integer at least 3, or there are fewer observations
            than the window.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, -0.005, 0.01, 0.003], index=idx)
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008, 0.018, 0.001, -0.004, 0.009, 0.002], index=idx)
        >>> rolling_alpha(a, b, window=5).dropna().shape[0]
        6
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)

    if not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3")
    validate_min_obs(a_aligned, min_obs=window, metric="rolling_alpha")

    rf_pp = _to_per_period_rf(risk_free, a_aligned.index, periods_per_year)
    y = (a_aligned - rf_pp).astype(float)
    x = (b_aligned - rf_pp).astype(float)
    x_with_const = sm.add_constant(x.to_frame(name="bench"), has_constant="add")

    model = RollingOLS(endog=y.to_numpy(), exog=x_with_const.to_numpy(), window=window)
    res = model.fit()
    alpha_pp = pd.Series(res.params[:, 0], index=a_aligned.index, name="alpha")
    if annualized:
        alpha_pp = alpha_pp * float(periods_per_year)
    return alpha_pp


def correlation(
    asset_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
) -> float:
    """Pearson correlation between asset and benchmark returns.

    Inner-aligns the two series on their common index before computing the
    sample Pearson correlation. Use this as a quick diagnostic alongside
    beta: beta = corr * sigma_a / sigma_b.

    Args:
        asset_returns: Periodic asset returns.
        benchmark_returns: Periodic benchmark returns; inner-aligned to the
            asset.

    Returns:
        Pearson correlation coefficient in ``[-1, 1]`` as a float.

    Raises:
        ValueError: If inputs are empty, contain NaN/inf, or have fewer
            than 2 overlapping observations.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> b = pd.Series([0.008, -0.015, 0.012, 0.003, -0.008])
        >>> -1.0 <= correlation(a, b) <= 1.0
        True
    """
    a = ensure_series(asset_returns, name="asset_returns")
    bm = ensure_series(benchmark_returns, name="benchmark_returns")
    a_aligned, b_aligned = align_inner(a, bm)
    validate_min_obs(a_aligned, min_obs=2, metric="correlation")
    return float(a_aligned.corr(b_aligned))
