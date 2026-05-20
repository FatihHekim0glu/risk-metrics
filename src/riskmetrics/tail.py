"""Tail-risk metrics: VaR (historical/parametric/Cornish-Fisher), CVaR/ES, skew, excess kurtosis, Jarque-Bera, tail ratio, Probabilistic Sharpe Ratio, Deflated Sharpe Ratio."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import ensure_series, validate_min_obs

__all__ = [
    "value_at_risk",
    "cornish_fisher_var",
    "conditional_value_at_risk",
    "skewness",
    "excess_kurtosis",
    "jarque_bera",
    "tail_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
]


def value_at_risk(
    returns: ReturnsLike,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value-at-Risk (VaR) at a given confidence level.

    Returns a NEGATIVE number representing the loss quantile: for daily SPY
    returns, the 95% historical VaR is approximately ``-0.02``, not ``+0.02``.
    The reported figure is the worst-case return that is not exceeded (to the
    downside) with probability ``confidence``.

    Args:
        returns: Periodic return series (Series, ndarray, or sequence).
        confidence: Confidence level in ``(0, 1)``. ``0.95`` selects the 5%
            lower tail.
        method: One of ``"historical"``, ``"parametric"`` / ``"gaussian"``, or
            ``"cornish_fisher"``.

            * ``"historical"`` uses the empirical lower quantile via
              ``np.quantile(..., method="lower")`` so the returned value is an
              actually observed return, never an interpolated point between
              two days.
            * ``"parametric"`` / ``"gaussian"`` assumes returns are normal and
              returns ``mean + std * Phi^{-1}(1 - confidence)``.
            * ``"cornish_fisher"`` delegates to :func:`cornish_fisher_var`.

    Returns:
        VaR as a negative float (loss expressed as a signed return).

    Raises:
        ValueError: If the input cannot be coerced to a Series, has fewer
            than 2 observations, or ``method`` is not recognized.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> value_at_risk(r, confidence=0.95, method="historical")
        -0.02
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="value_at_risk")

    if method == "historical":
        return float(np.quantile(r.to_numpy(), 1.0 - confidence, method="lower"))
    if method in ("parametric", "gaussian"):
        return float(r.mean() + r.std(ddof=1) * stats.norm.ppf(1.0 - confidence))
    if method == "cornish_fisher":
        return cornish_fisher_var(r, confidence=confidence)
    raise ValueError(
        f"Unknown method {method!r}; expected one of "
        "'historical', 'parametric', 'gaussian', 'cornish_fisher'"
    )


def cornish_fisher_var(
    returns: ReturnsLike,
    confidence: float = 0.95,
) -> float:
    """Cornish-Fisher Value-at-Risk adjusted for skewness and excess kurtosis.

    Applies the Cornish-Fisher quantile expansion to correct the Gaussian VaR
    for higher-order moments of the return distribution. The expansion is
    accurate near the centre of the distribution but degrades for very fat
    tails or strong asymmetry; a :class:`UserWarning` is emitted when the
    sample moments leave the regime ``|skew| <= 1`` and ``excess kurt <= 5``.

    Formula:

        z      = Phi^{-1}(1 - confidence)
        S      = sample skewness (bias-corrected)
        K      = sample EXCESS kurtosis (Fisher, bias-corrected)
        z_cf   = z + (z^2 - 1) * S / 6
                   + (z^3 - 3z) * K / 24
                   - (2 z^3 - 5 z) * S^2 / 36
        VaR    = mean + std * z_cf

    Args:
        returns: Periodic return series.
        confidence: Confidence level in ``(0, 1)``.

    Returns:
        Cornish-Fisher VaR as a (typically negative) float.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Warns:
        UserWarning: If ``abs(skew) > 1`` or ``excess kurt > 5``, signalling
            that the expansion is outside its reliable domain.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> cf = cornish_fisher_var(r, confidence=0.95)
        >>> isinstance(cf, float)
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="cornish_fisher_var")

    arr = r.to_numpy()
    z = stats.norm.ppf(1.0 - confidence)
    s_skew = float(stats.skew(arr, bias=False))
    k_excess = float(stats.kurtosis(arr, fisher=True, bias=False))

    if abs(s_skew) > 1.0 or k_excess > 5.0:
        warnings.warn(
            "Cornish-Fisher expansion may be unreliable outside |S|<1, K<5 regime",
            UserWarning,
            stacklevel=2,
        )

    z_cf = (
        z
        + (z ** 2 - 1.0) * s_skew / 6.0
        + (z ** 3 - 3.0 * z) * k_excess / 24.0
        - (2.0 * z ** 3 - 5.0 * z) * s_skew ** 2 / 36.0
    )
    return float(r.mean() + r.std(ddof=1) * z_cf)


def conditional_value_at_risk(
    returns: ReturnsLike,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Conditional Value-at-Risk (CVaR), also known as Expected Shortfall.

    The average loss conditional on the loss exceeding VaR. CVaR is a coherent
    risk measure (subadditive), whereas VaR is not. Like :func:`value_at_risk`,
    the result is reported as a NEGATIVE number.

    Args:
        returns: Periodic return series.
        confidence: Confidence level in ``(0, 1)``.
        method: One of ``"historical"`` or ``"parametric"`` / ``"gaussian"``.

            * ``"historical"`` averages the returns at or below the historical
              VaR. If no observation strictly satisfies ``r <= VaR`` (degenerate
              single-observation case), the VaR itself is returned.
            * ``"parametric"`` / ``"gaussian"`` uses the closed-form Gaussian
              ES: ``mean - std * phi(z) / (1 - confidence)`` where ``z`` is the
              standard-normal lower quantile.

    Returns:
        CVaR as a (typically negative) float.

    Raises:
        ValueError: If the input cannot be coerced to a Series, has fewer
            than 2 observations, or ``method`` is not recognized.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> conditional_value_at_risk(r, confidence=0.95, method="historical")
        -0.02
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="conditional_value_at_risk")

    if method == "historical":
        var = value_at_risk(r, confidence=confidence, method="historical")
        tail = r[r <= var]
        if len(tail) == 0:
            return float(var)
        return float(tail.mean())
    if method in ("parametric", "gaussian"):
        z = stats.norm.ppf(1.0 - confidence)
        return float(
            r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1.0 - confidence)
        )
    raise ValueError(
        f"Unknown method {method!r}; expected one of "
        "'historical', 'parametric', 'gaussian'"
    )


def skewness(returns: ReturnsLike, bias: bool = False) -> float:
    """Sample skewness of a return series.

    Thin wrapper around :func:`scipy.stats.skew`. The default ``bias=False``
    selects the small-sample bias-corrected G1 estimator (the same convention
    used by Excel's ``SKEW`` and ``pandas.Series.skew``).

    Args:
        returns: Periodic return series.
        bias: If False (default), apply the G1 small-sample bias correction.
            If True, return the biased moment ratio.

    Returns:
        Sample skewness as a float. Negative values indicate a left-skewed
        (fat downside) distribution.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> isinstance(skewness(r), float)
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="skewness")
    return float(stats.skew(r.to_numpy(), bias=bias))


def excess_kurtosis(returns: ReturnsLike, bias: bool = False) -> float:
    """Excess (Fisher) kurtosis of a return series.

    Returns kurtosis on the FISHER convention: a normal distribution has
    excess kurtosis of 0 (NOT 3). Positive values indicate fatter tails than
    the Gaussian; values above 3-5 imply parametric VaR is likely to
    understate tail risk.

    Args:
        returns: Periodic return series.
        bias: If False (default), apply the small-sample bias correction.

    Returns:
        Excess kurtosis as a float (normal ~ 0).

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> isinstance(excess_kurtosis(r), float)
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="excess_kurtosis")
    return float(stats.kurtosis(r.to_numpy(), fisher=True, bias=bias))


def jarque_bera(returns: ReturnsLike) -> tuple[float, float]:
    """Jarque-Bera test for normality of returns.

    The null hypothesis is that the returns are drawn from a normal
    distribution. A small p-value rejects normality and is a red flag for
    parametric VaR / Gaussian ES, which assume normal returns.

    Args:
        returns: Periodic return series.

    Returns:
        Tuple ``(JB_stat, p_value)``: the Jarque-Bera test statistic
        (chi-squared, 2 dof) and its p-value.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> stat, p = jarque_bera(r)
        >>> isinstance(stat, float) and isinstance(p, float)
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="jarque_bera")
    result = stats.jarque_bera(r.to_numpy())
    return float(result.statistic), float(result.pvalue)


def tail_ratio(returns: ReturnsLike, percentile: float = 0.05) -> float:
    """Ratio of the right-tail magnitude to the left-tail magnitude.

    Defined as ``|Q(1 - p)| / |Q(p)|`` where ``Q`` is the empirical quantile
    function and ``p`` is the tail percentile (default 5%). Values greater
    than 1 indicate that big up-moves dominate big down-moves (right-skewed
    payoffs); values below 1 mean the left tail is heavier in magnitude.

    Args:
        returns: Periodic return series.
        percentile: Tail probability in ``(0, 0.5)``. Default 0.05 contrasts
            the 95th and 5th percentiles.

    Returns:
        Tail ratio as a non-negative float. Infinity if the left-tail quantile
        is exactly zero.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> isinstance(tail_ratio(r, percentile=0.2), float)
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="tail_ratio")
    arr = r.to_numpy()
    right = abs(float(np.quantile(arr, 1.0 - percentile)))
    left = abs(float(np.quantile(arr, percentile)))
    if left == 0.0:
        return float("inf")
    return right / left


def probabilistic_sharpe_ratio(
    returns: ReturnsLike,
    threshold_sr: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) of Bailey & Lopez de Prado (2012).

    Estimates the probability that the true (population) Sharpe Ratio of the
    strategy exceeds a benchmark Sharpe ``threshold_sr``, given the observed
    sample Sharpe and the higher-moment evidence about non-normality. PSR
    explicitly penalises strategies whose track records are short, negatively
    skewed, or fat-tailed.

    Computation works in per-period units throughout. The annual benchmark
    Sharpe is rescaled to a per-period benchmark via
    ``threshold_sr / sqrt(periods_per_year)``, and the observed per-period
    Sharpe is compared against it. The variance of the Sharpe estimator uses
    the FULL (Pearson) kurtosis ``K_full = stats.kurtosis(..., fisher=False)``
    -- using excess kurtosis here is a common bug.

    Args:
        returns: Periodic return series.
        threshold_sr: Annualized benchmark Sharpe to beat. Default 0 tests
            whether the strategy's true Sharpe is positive.
        periods_per_year: Periods per year used to rescale ``threshold_sr``
            from annual to per-period units (e.g. 252 for daily data).

    Returns:
        Probability in ``[0, 1]`` that the true Sharpe exceeds the threshold.

    Raises:
        ValueError: If the input cannot be coerced to a Series or has fewer
            than 2 observations.

    References:
        Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio
        Efficient Frontier. Journal of Risk, 15(2).

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> 0.0 <= probabilistic_sharpe_ratio(r) <= 1.0
        True
    """
    r = ensure_series(returns)
    validate_min_obs(r, min_obs=2, metric="probabilistic_sharpe_ratio")

    arr = r.to_numpy()
    std = float(r.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        warnings.warn(
            "zero volatility; probabilistic Sharpe undefined",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    sr_obs = float(r.mean()) / std
    sr_star = float(threshold_sr) / float(np.sqrt(periods_per_year))
    # scipy emits RuntimeWarning on near-constant inputs (m2**2 underflow); the
    # downstream NaN/finiteness guards turn that into a clean NaN return.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        s_skew = float(stats.skew(arr, bias=False))
        k_full = float(stats.kurtosis(arr, fisher=False, bias=False))
    n = len(r)

    if not (np.isfinite(s_skew) and np.isfinite(k_full)):
        return float("nan")

    denom_sq = 1.0 - s_skew * sr_obs + (k_full - 1.0) / 4.0 * sr_obs ** 2
    if denom_sq <= 0.0 or not np.isfinite(denom_sq):
        return float("nan")
    denom = float(np.sqrt(denom_sq))
    z = (sr_obs - sr_star) * float(np.sqrt(n - 1)) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: ReturnsLike,
    n_trials: int,
    sr_variance: float,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Deflated Sharpe Ratio (DSR) of Bailey & Lopez de Prado (2014).

    Adjusts the Probabilistic Sharpe Ratio for selection bias: when ``N``
    strategy variants are backtested and the best is reported, the expected
    maximum Sharpe under the null is far above zero. DSR computes that
    expected maximum and feeds it to :func:`probabilistic_sharpe_ratio` as the
    benchmark.

    The expected-max threshold (annualized) is:

        SR* = sqrt(V) * [(1 - gamma) * Phi^{-1}(1 - 1/N)
                          + gamma * Phi^{-1}(1 - 1/(N e))]

    where ``V`` is the cross-trial variance of the Sharpe estimates and
    ``gamma`` is the Euler-Mascheroni constant.

    HONESTY REQUIREMENT: ``n_trials`` must reflect every strategy variant
    actually evaluated -- including every parameter sweep, every feature
    combination, every threshold tweak. Under-reporting ``n_trials`` defeats
    the entire point of DSR. Keep a log.

    Args:
        returns: Periodic return series of the SELECTED strategy.
        n_trials: Number of independent strategy variants that were tested
            before selection. Must be at least 2.
        sr_variance: Variance of the Sharpe-ratio estimates across the
            ``n_trials`` candidates. Annualized units, matching
            ``threshold_sr`` in :func:`probabilistic_sharpe_ratio`.
        periods_per_year: Periods per year used to rescale the threshold.

    Returns:
        Probability in ``[0, 1]`` that the true Sharpe exceeds the
        selection-bias-adjusted threshold.

    Raises:
        ValueError: If the input cannot be coerced to a Series, has fewer
            than 2 observations, ``n_trials < 2``, or ``sr_variance < 0``.

    References:
        Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting, and
        Non-Normality. Journal of Portfolio Management, 40(5).

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> dsr = deflated_sharpe_ratio(r, n_trials=10, sr_variance=0.5)
        >>> 0.0 <= dsr <= 1.0
        True
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2, got {n_trials}")
    if sr_variance < 0.0:
        raise ValueError(f"sr_variance must be non-negative, got {sr_variance}")

    gamma = 0.5772156649  # Euler-Mascheroni
    sr_star_annual = float(np.sqrt(sr_variance)) * (
        (1.0 - gamma) * float(stats.norm.ppf(1.0 - 1.0 / n_trials))
        + gamma * float(stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    )
    return probabilistic_sharpe_ratio(
        returns,
        threshold_sr=sr_star_annual,
        periods_per_year=periods_per_year,
    )
