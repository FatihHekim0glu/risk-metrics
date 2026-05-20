"""One-shot tearsheet aggregator: every public metric collapsed into a single labeled Series.
Grouped by ``returns.*``, ``volatility.*``, ``drawdown.*``, ``ratios.*``, ``tail.*``, and ``benchmark.*`` so callers can slice by prefix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics.drawdown import (
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    pain_index,
    sterling_ratio,
    ulcer_index,
)
from riskmetrics.ratios import (
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from riskmetrics.returns import (
    annualized_return_arithmetic,
    best_period,
    cagr,
    hit_rate,
    total_return,
    worst_period,
)
from riskmetrics.volatility import (
    downside_deviation,
    mean_absolute_deviation,
    realized_volatility,
    semi_deviation,
)


def _safe(fn, *args, **kwargs):
    """Call ``fn`` and return its result, or ``np.nan`` on any exception.

    Tearsheet rows must never fail the whole call; individual metrics may be
    undefined (e.g. zero-vol Sharpe, no-drawdown Calmar) and we just record NaN.
    """
    try:
        out = fn(*args, **kwargs)
    except Exception:
        return float("nan")
    if isinstance(out, tuple):
        # ``best_period``/``worst_period`` return (label, value); the value is what we tabulate.
        return float(out[1]) if np.isfinite(out[1]) else float("nan")
    if isinstance(out, (int, float, np.floating, np.integer)):
        return float(out)
    return out


def tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Aggregate every library metric into one labeled ``pd.Series``.

    The index is dotted-prefix metric names (``returns.cagr``, ``volatility.annualized``,
    ``drawdown.max``, ``ratios.sharpe``, ``tail.var_95_historical``,
    ``benchmark.alpha_annualized``, etc.). Group order is stable so callers can slice
    by prefix (``s.filter(like="ratios.")``).

    Args:
        returns: Periodic simple returns indexed by a ``DatetimeIndex``.
        benchmark: Optional periodic returns of a benchmark, aligned to ``returns``.
            If ``None``, benchmark-relative rows (alpha/beta/IR/up-down capture and
            ``ratios.m_squared``) are omitted.
        risk_free: Annual risk-free rate as scalar or annual-rate Series.
        periods_per_year: Annualisation factor (252 for daily, 12 for monthly).

    Returns:
        ``pd.Series`` of floats with metric names as the index. Failures collapse
        to ``NaN`` rather than raising, so the call is safe to drop straight into
        a dashboard.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=300, freq="B")
        >>> r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300), index=idx)
        >>> s = tearsheet(r)
        >>> "returns.cagr" in s.index
        True
    """
    rows: dict[str, float] = {}

    # ---- returns.* -------------------------------------------------------------
    rows["returns.total_return"] = _safe(total_return, returns)
    rows["returns.cagr"] = _safe(cagr, returns, periods_per_year=periods_per_year)
    rows["returns.annualized_arithmetic"] = _safe(
        annualized_return_arithmetic, returns, periods_per_year=periods_per_year
    )
    rows["returns.best_period"] = _safe(best_period, returns)
    rows["returns.worst_period"] = _safe(worst_period, returns)
    rows["returns.hit_rate"] = _safe(hit_rate, returns)

    # ---- volatility.* ----------------------------------------------------------
    rows["volatility.annualized"] = _safe(
        realized_volatility, returns, periods_per_year=periods_per_year
    )
    rows["volatility.downside_deviation"] = _safe(
        downside_deviation, returns, periods_per_year=periods_per_year
    )
    rows["volatility.semi_deviation"] = _safe(
        semi_deviation, returns, periods_per_year=periods_per_year
    )
    rows["volatility.mad"] = _safe(
        mean_absolute_deviation, returns, periods_per_year=periods_per_year
    )

    # ---- drawdown.* ------------------------------------------------------------
    rows["drawdown.max"] = _safe(max_drawdown, returns)
    rows["drawdown.duration_max_days"] = _safe(max_drawdown_duration, returns)
    rows["drawdown.ulcer_index"] = _safe(ulcer_index, returns)
    rows["drawdown.pain_index"] = _safe(pain_index, returns)
    rows["drawdown.calmar"] = _safe(
        calmar_ratio, returns, periods_per_year=periods_per_year
    )
    rows["drawdown.sterling"] = _safe(
        sterling_ratio, returns, periods_per_year=periods_per_year
    )

    # ---- ratios.* --------------------------------------------------------------
    rows["ratios.sharpe"] = _safe(
        sharpe_ratio,
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        smart=False,
    )
    rows["ratios.sharpe_smart"] = _safe(
        sharpe_ratio,
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        smart=True,
    )
    rows["ratios.sortino"] = _safe(
        sortino_ratio,
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
    )
    rows["ratios.omega"] = _safe(omega_ratio, returns)

    # ---- tail.* ----------------------------------------------------------------
    # The dedicated ``tail`` module may not be wired up yet; fall back to inline
    # implementations so the tearsheet stays usable in CI before ``tail.py`` lands.
    try:
        from riskmetrics import tail as _tail  # type: ignore[attr-defined]
    except ImportError:
        _tail = None  # type: ignore[assignment]

    if _tail is not None:
        rows["tail.var_95_historical"] = _safe(
            _tail.value_at_risk, returns, confidence=0.95, method="historical"
        )
        rows["tail.var_99_historical"] = _safe(
            _tail.value_at_risk, returns, confidence=0.99, method="historical"
        )
        rows["tail.var_95_cornish_fisher"] = _safe(
            _tail.cornish_fisher_var, returns, confidence=0.95
        )
        rows["tail.cvar_95"] = _safe(
            _tail.conditional_value_at_risk, returns, confidence=0.95, method="historical"
        )
        rows["tail.cvar_99"] = _safe(
            _tail.conditional_value_at_risk, returns, confidence=0.99, method="historical"
        )
        rows["tail.skew"] = _safe(_tail.skewness, returns)
        rows["tail.excess_kurtosis"] = _safe(_tail.excess_kurtosis, returns)
        # jarque_bera returns (stat, p_value); _safe() unpacks tuples to [1].
        rows["tail.jarque_bera_pvalue"] = _safe(_tail.jarque_bera, returns)
        rows["tail.psr_vs_zero"] = _safe(
            _tail.probabilistic_sharpe_ratio, returns, threshold_sr=0.0
        )
    else:
        from scipy import stats as _sps

        try:
            r = pd.Series(returns).dropna()
            arr = r.to_numpy()
            rows["tail.var_95_historical"] = float(-np.quantile(arr, 0.05))
            rows["tail.var_99_historical"] = float(-np.quantile(arr, 0.01))
            mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
            s = float(_sps.skew(arr, bias=False))
            k = float(_sps.kurtosis(arr, fisher=True, bias=False))
            z = _sps.norm.ppf(0.05)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (
                2 * z**3 - 5 * z
            ) * s**2 / 36
            rows["tail.var_95_cornish_fisher"] = float(-(mu + sigma * z_cf))
            rows["tail.cvar_95"] = float(-arr[arr <= np.quantile(arr, 0.05)].mean())
            rows["tail.cvar_99"] = float(-arr[arr <= np.quantile(arr, 0.01)].mean())
            rows["tail.skew"] = s
            rows["tail.excess_kurtosis"] = k
            _, jb_p = _sps.jarque_bera(arr)
            rows["tail.jarque_bera_pvalue"] = float(jb_p)
            sr = float(
                sharpe_ratio(
                    returns,
                    risk_free=risk_free,
                    periods_per_year=periods_per_year,
                    smart=False,
                )
            )
            n = len(arr)
            psr_num = sr * np.sqrt(n - 1)
            psr_den = np.sqrt(1 - s * sr + (k / 4) * sr**2)
            rows["tail.psr_vs_zero"] = (
                float(_sps.norm.cdf(psr_num / psr_den)) if psr_den > 0 else float("nan")
            )
        except Exception:
            for key in (
                "tail.var_95_historical",
                "tail.var_99_historical",
                "tail.var_95_cornish_fisher",
                "tail.cvar_95",
                "tail.cvar_99",
                "tail.skew",
                "tail.excess_kurtosis",
                "tail.jarque_bera_pvalue",
                "tail.psr_vs_zero",
            ):
                rows[key] = float("nan")

    # ---- benchmark.* (only when a benchmark is supplied) ----------------------
    if benchmark is not None:
        try:
            from riskmetrics import benchmark as _bench  # type: ignore[attr-defined]
        except ImportError:
            _bench = None  # type: ignore[assignment]

        if _bench is not None:
            # alpha_annualized / alpha_tstat / r_squared all come from the same
            # OLS fit; compute it once and unpack the dataclass.
            capm_nan = (float("nan"), float("nan"), float("nan"))
            try:
                capm = _bench.alpha_beta(
                    returns,
                    benchmark,
                    risk_free=risk_free,
                    periods_per_year=periods_per_year,
                )
                alpha_ann = float(capm.alpha_annualized)
                alpha_t = float(capm.alpha_tstat)
                r2 = float(capm.r_squared)
            except Exception:
                alpha_ann, alpha_t, r2 = capm_nan
            rows["benchmark.alpha_annualized"] = alpha_ann
            rows["benchmark.alpha_tstat"] = alpha_t
            rows["benchmark.beta"] = _safe(_bench.beta, returns, benchmark)
            rows["benchmark.r_squared"] = r2
            rows["benchmark.tracking_error"] = _safe(
                _bench.tracking_error,
                returns,
                benchmark,
                periods_per_year=periods_per_year,
            )
            rows["benchmark.information_ratio"] = _safe(
                _bench.information_ratio,
                returns,
                benchmark,
                periods_per_year=periods_per_year,
            )
            rows["benchmark.up_capture"] = _safe(
                _bench.up_capture, returns, benchmark, periods_per_year=periods_per_year
            )
            rows["benchmark.down_capture"] = _safe(
                _bench.down_capture, returns, benchmark, periods_per_year=periods_per_year
            )
            rows["benchmark.correlation"] = _safe(
                _bench.correlation, returns, benchmark
            )
            # ``ratios.m_squared`` depends on benchmark vol, so it lives down here.
            try:
                bench_vol = float(
                    realized_volatility(benchmark, periods_per_year=periods_per_year)
                )
                from riskmetrics.ratios import m_squared

                rows["ratios.m_squared"] = _safe(
                    m_squared,
                    returns,
                    benchmark_vol_annualized=bench_vol,
                    risk_free=risk_free,
                    periods_per_year=periods_per_year,
                )
            except Exception:
                rows["ratios.m_squared"] = float("nan")
        else:
            # ``statsmodels`` (or the not-yet-built benchmark module) is unavailable.
            # Compute the closed-form pieces we can without OLS so dashboards still get something.
            try:
                r_al, b_al = returns.align(benchmark, join="inner")
                r_al = r_al.dropna()
                b_al = b_al.loc[r_al.index]
                cov = float(np.cov(r_al, b_al, ddof=1)[0, 1])
                var_b = float(np.var(b_al, ddof=1))
                beta = cov / var_b if var_b > 0 else float("nan")
                corr = float(np.corrcoef(r_al, b_al)[0, 1])
                te = float(
                    (r_al - b_al).std(ddof=1) * np.sqrt(periods_per_year)
                )
                ir_num = float((r_al - b_al).mean() * periods_per_year)
                ir = ir_num / te if te > 0 else float("nan")
                up_mask = b_al > 0
                dn_mask = b_al < 0
                up_cap = (
                    float(r_al[up_mask].mean() / b_al[up_mask].mean())
                    if up_mask.any() and b_al[up_mask].mean() != 0
                    else float("nan")
                )
                dn_cap = (
                    float(r_al[dn_mask].mean() / b_al[dn_mask].mean())
                    if dn_mask.any() and b_al[dn_mask].mean() != 0
                    else float("nan")
                )
                rows["benchmark.alpha_annualized"] = float("nan")
                rows["benchmark.alpha_tstat"] = float("nan")
                rows["benchmark.beta"] = beta
                rows["benchmark.r_squared"] = corr * corr
                rows["benchmark.tracking_error"] = te
                rows["benchmark.information_ratio"] = ir
                rows["benchmark.up_capture"] = up_cap
                rows["benchmark.down_capture"] = dn_cap
                rows["benchmark.correlation"] = corr
                bench_vol = float(
                    realized_volatility(benchmark, periods_per_year=periods_per_year)
                )
                from riskmetrics.ratios import m_squared

                rows["ratios.m_squared"] = _safe(
                    m_squared,
                    returns,
                    benchmark_vol_annualized=bench_vol,
                    risk_free=risk_free,
                    periods_per_year=periods_per_year,
                )
            except Exception:
                for key in (
                    "benchmark.alpha_annualized",
                    "benchmark.alpha_tstat",
                    "benchmark.beta",
                    "benchmark.r_squared",
                    "benchmark.tracking_error",
                    "benchmark.information_ratio",
                    "benchmark.up_capture",
                    "benchmark.down_capture",
                    "benchmark.correlation",
                    "ratios.m_squared",
                ):
                    rows[key] = float("nan")

    series = pd.Series(rows, dtype=float, name="tearsheet")
    series.index.name = "metric"
    return series
