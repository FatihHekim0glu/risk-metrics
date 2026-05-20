from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from riskmetrics import __version__

# ---- Page setup & palette --------------------------------------------------
st.set_page_config(page_title="Risk metrics", layout="wide")

# Okabe-Ito (deuteranope-safe). Portfolio = blue, benchmark = orange.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish-green
    "#CC79A7",  # reddish-purple
    "#56B4E9",  # sky-blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]
PORTFOLIO_COLOR = "#0072B2"
BENCHMARK_COLOR = "#E69F00"
LOSS_COLOR = "#D55E00"
GAIN_COLOR = "#009E73"

px.defaults.color_discrete_sequence = OKABE_ITO


# ---- Cached helpers --------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching prices...")
def fetch(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Cached price fetch wrapping ``riskmetrics.data.get_prices``."""
    from riskmetrics.data import get_prices

    return get_prices(list(tickers), start=start, end=end)


@st.cache_data(ttl=3600, show_spinner="Fetching risk-free rate...")
def fetch_rf(start: str, end: str) -> pd.Series:
    """Cached FRED DGS3MO pull, returned as a per-day rate series."""
    from riskmetrics.data import get_risk_free

    return get_risk_free(start=start, end=end, series="DGS3MO")


# ---- Sidebar ---------------------------------------------------------------
st.title("Risk metrics dashboard")
st.caption(f"Built with riskmetrics v{__version__}")

with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_area("Tickers (one per line)", value="SPY\nAAPL")
    weights_raw = st.text_area(
        "Weights (one per line, must sum to 1.0)", value="0.5\n0.5"
    )
    bench_ticker = st.text_input("Benchmark ticker", value="SPY")
    date_range = st.date_input(
        "Date range", value=(date(2020, 1, 1), date(2024, 12, 31))
    )
    use_fred_rf = st.checkbox("Use FRED risk-free rate (DGS3MO)", value=True)
    run = st.button("Run analysis", type="primary")


def _parse_tokens(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(float(line))
    return out


# ---- Main ------------------------------------------------------------------
if not run:
    st.info(
        "Enter tickers, weights, benchmark and a date range in the sidebar, "
        "then click **Run analysis**."
    )
    st.stop()

# Parse + validate inputs.
try:
    tickers = _parse_tokens(tickers_raw)
    weights = _parse_floats(weights_raw)
except ValueError as exc:
    st.error(f"Could not parse weights as floats: {exc}")
    st.stop()

if not tickers:
    st.error("Provide at least one ticker.")
    st.stop()
if len(tickers) != len(weights):
    st.error(
        f"Got {len(tickers)} tickers but {len(weights)} weights; counts must match."
    )
    st.stop()
weight_sum = sum(weights)
if not np.isclose(weight_sum, 1.0, atol=1e-6):
    st.error(f"Weights must sum to 1.0 (got {weight_sum:.6f}).")
    st.stop()
if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.error("Pick a start *and* end date.")
    st.stop()
start_d, end_d = date_range
if start_d >= end_d:
    st.error("Start date must be strictly before end date.")
    st.stop()

start_s = start_d.isoformat()
end_s = end_d.isoformat()

# All tickers needed: portfolio + benchmark (deduped).
all_tickers = tuple(dict.fromkeys([*tickers, bench_ticker]))

with st.spinner("Computing metrics..."):
    prices = fetch(all_tickers, start_s, end_s)

if prices.empty:
    st.error("No price data returned for the requested tickers/window.")
    st.stop()

# Per-ticker simple returns; align on intersection.
rets = prices.pct_change().dropna(how="all")
missing_per_ticker = rets.isna().sum().to_dict()
coverage = (1.0 - rets.isna().mean()).to_dict()

# Drop any column with no data for the portfolio computation, but keep diagnostics.
rets_aligned = rets[list(tickers)].dropna()
if rets_aligned.empty:
    st.error(
        "No overlapping non-NaN return observations across the requested portfolio tickers."
    )
    st.stop()

w = np.asarray(weights, dtype=float)
portfolio_returns = pd.Series(rets_aligned.to_numpy() @ w, index=rets_aligned.index)
portfolio_returns.name = "portfolio"

if bench_ticker in rets.columns:
    benchmark_returns = rets[bench_ticker].dropna()
    benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()
    portfolio_returns = portfolio_returns.loc[benchmark_returns.index]
else:
    benchmark_returns = pd.Series(dtype=float)
    st.warning(f"Benchmark ticker {bench_ticker!r} returned no data.")

# Risk-free.
if use_fred_rf:
    try:
        rf_daily = fetch_rf(start_s, end_s)
        # ``get_risk_free`` returns per-day rates; convert back to annual for the
        # ratios API (it expects an annual rate that it then geometrically de-annualises).
        rf_annual = (1.0 + rf_daily) ** 252 - 1.0
        rf_annual = rf_annual.reindex(portfolio_returns.index).ffill().bfill()
        risk_free_arg: float | pd.Series = rf_annual
    except Exception as exc:
        st.warning(f"FRED fetch failed ({exc}); using 0% risk-free.")
        risk_free_arg = 0.0
else:
    risk_free_arg = 0.0

# Compute tearsheet.
from riskmetrics.tearsheet import tearsheet as _tearsheet

ts_portfolio = _tearsheet(
    portfolio_returns,
    benchmark=benchmark_returns if not benchmark_returns.empty else None,
    risk_free=risk_free_arg,
    periods_per_year=252,
)
ts_benchmark = (
    _tearsheet(benchmark_returns, risk_free=risk_free_arg, periods_per_year=252)
    if not benchmark_returns.empty
    else None
)


def _g(s: pd.Series | None, key: str) -> float:
    """Safe getter on a tearsheet Series."""
    if s is None or key not in s.index:
        return float("nan")
    val = s.loc[key]
    return float(val) if pd.notna(val) else float("nan")


def _fmt_pct(x: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}"


# ---- Hero strip ------------------------------------------------------------
hero = st.columns(5)
hero_metrics = [
    ("CAGR", "returns.cagr", _fmt_pct, True),
    ("Annualized vol", "volatility.annualized", _fmt_pct, False),
    ("Sharpe", "ratios.sharpe", _fmt_num, True),
    ("Max drawdown", "drawdown.max", _fmt_pct, True),
    ("95% historical VaR", "tail.var_95_historical", _fmt_pct, False),
]
for col, (label, key, fmt, higher_is_better) in zip(hero, hero_metrics, strict=True):
    p = _g(ts_portfolio, key)
    b = _g(ts_benchmark, key)
    if np.isfinite(p) and np.isfinite(b):
        diff = p - b
        delta_str = fmt(diff)
        # For drawdown and vol/VaR, "higher" is worse.
        delta_color = (
            "normal"
            if higher_is_better
            else ("inverse" if label in {"Max drawdown"} else "inverse")
        )
        col.metric(label, fmt(p), delta=delta_str, delta_color=delta_color)
    else:
        col.metric(label, fmt(p))


# ---- Auto-diagnostics ------------------------------------------------------
warnings_list: list[str] = []
n_obs = int(len(portfolio_returns))
arr = portfolio_returns.to_numpy()

# Auto-diagnostics fire ONLY when actionable: the test surfaces a number the
# user should re-interpret a metric by, not just a statistical curiosity. Daily
# equity returns are reliably non-Gaussian, autocorrelated, and fat-tailed --
# the goal here is to warn when those properties materially distort the
# headline numbers, not to echo well-known stylised facts.

# 1. Jarque-Bera + VaR gap: flag only if parametric VaR understates historical
# VaR by more than 20% (the level at which the choice of VaR method matters
# for risk budgeting). Non-normality without a material gap is benign here.
try:
    _, jb_p = stats.jarque_bera(arr)
    hist_var = _g(ts_portfolio, "tail.var_95_historical")
    mu_p, sd_p = float(arr.mean()), float(arr.std(ddof=1))
    param_var = -(mu_p + sd_p * stats.norm.ppf(0.05))
    if (
        jb_p < 1e-6
        and np.isfinite(hist_var)
        and hist_var > 0
        and np.isfinite(param_var)
    ):
        pct_off = (hist_var - param_var) / hist_var * 100
        if pct_off > 20.0:
            warnings_list.append(
                f"Parametric 95% VaR understates the historical 95% VaR by "
                f"{pct_off:.0f}% (JB p={jb_p:.1e}). Prefer historical or "
                f"Cornish-Fisher VaR for tail-risk budgeting."
            )
except Exception:
    pass

# 2. Ljung-Box + Sharpe sensitivity: flag only if the autocorrelation-adjusted
# Sharpe differs from the naive Sharpe by more than 10% in absolute terms.
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_res = acorr_ljungbox(portfolio_returns, lags=[5], return_df=True)
    lb_p = float(lb_res["lb_pvalue"].iloc[0])
    naive_sr = _g(ts_portfolio, "ratios.sharpe")
    smart_sr = _g(ts_portfolio, "ratios.sharpe_smart")
    if (
        lb_p < 1e-3
        and np.isfinite(naive_sr)
        and np.isfinite(smart_sr)
        and abs(naive_sr) > 1e-6
        and abs(naive_sr - smart_sr) / abs(naive_sr) > 0.10
    ):
        warnings_list.append(
            f"Serial correlation materially inflates Sharpe: naive "
            f"{naive_sr:.2f} vs autocorrelation-adjusted {smart_sr:.2f} "
            f"(Ljung-Box p={lb_p:.1e})."
        )
except Exception:
    pass

# 3. Small sample
if n_obs < 252:
    se = 1.0 / np.sqrt(n_obs) if n_obs > 0 else float("nan")
    warnings_list.append(
        f"Sample size {n_obs} (< 252) — Sharpe standard error ≈ {se:.2f}; "
        f"treat the headline figure as indicative."
    )

# 4. Open drawdown at end of window: only flag when the open drawdown is at
# least 10% deep (a portfolio finishing 2% below a peak isn't notable).
try:
    from riskmetrics.drawdown import drawdown_table

    dd_tbl = drawdown_table(portfolio_returns, top=10)
    open_eps = dd_tbl[dd_tbl["is_open"]] if not dd_tbl.empty else dd_tbl
    if not open_eps.empty:
        worst_open = open_eps.sort_values("drawdown").iloc[0]
        if float(worst_open["drawdown"]) <= -0.10:
            warnings_list.append(
                f"Open drawdown {worst_open['drawdown']:.1%} from peak "
                f"{pd.Timestamp(worst_open['peak_date']).date()} has not "
                f"recovered as of {portfolio_returns.index[-1].date()}; "
                f"reported duration is right-censored."
            )
except Exception:
    pass

# 5. Rolling 90-day beta instability: only flag when the beta range exceeds
# 1.0 across the sample (a swing of one full unit is genuinely unstable).
if not benchmark_returns.empty:
    try:
        joined = pd.concat(
            [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
            axis=1,
            join="inner",
        ).dropna()
        if len(joined) >= 90:
            cov = joined["p"].rolling(90).cov(joined["b"])
            var_b = joined["b"].rolling(90).var()
            roll_beta = (cov / var_b).dropna()
            if not roll_beta.empty:
                lo, hi = float(roll_beta.min()), float(roll_beta.max())
                if hi - lo > 1.0:
                    warnings_list.append(
                        f"Rolling 90-day beta ranges {lo:.2f}–{hi:.2f} "
                        f"(> 1.0 swing); single-beta CAPM alpha is unreliable."
                    )
    except Exception:
        pass

# 6. Skewness: only flag |skew| > 2 (extreme asymmetry; |skew| around 0.5 is
# normal for equity).
try:
    s_val = float(stats.skew(arr, bias=False))
    if abs(s_val) > 2.0:
        warnings_list.append(
            f"Returns are extremely skewed (S={s_val:.2f}); the symmetry "
            f"assumption of Sharpe and parametric VaR is suspect."
        )
except Exception:
    pass

# 7. Excess kurtosis: only flag >10 (well above the 3-5 typical for daily
# equity).
try:
    k_val = float(stats.kurtosis(arr, fisher=True, bias=False))
    if k_val > 10.0:
        warnings_list.append(
            f"Returns have exceptional fat tails (excess kurt = {k_val:.1f}); "
            f"prefer historical VaR over parametric or Cornish-Fisher."
        )
except Exception:
    pass

n_warn = len(warnings_list)
expander_label = (
    f"⚠ {n_warn} model warnings — expand"
    if n_warn > 0
    else "0 model warnings — expand"
)
with st.expander(expander_label, expanded=(n_warn > 0)):
    if not warnings_list:
        st.success("No diagnostic flags triggered.")
    for msg in warnings_list:
        st.warning(msg)


# ---- Tabs ------------------------------------------------------------------
tab_perf, tab_risk, tab_radj, tab_tail, tab_bench, tab_diag = st.tabs(
    [
        "Performance",
        "Risk profile",
        "Risk-adjusted",
        "Tail risk",
        "Benchmark",
        "Diagnostics",
    ]
)

# ---- Performance tab -------------------------------------------------------
with tab_perf:
    st.subheader("Equity curve (log scale)")
    eq_p = (1 + portfolio_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_p.index,
            y=eq_p.values,
            mode="lines",
            name="Portfolio",
            line={"color": PORTFOLIO_COLOR, "width": 2},
        )
    )
    if not benchmark_returns.empty:
        eq_b = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=eq_b.index,
                y=eq_b.values,
                mode="lines",
                name=bench_ticker,
                line={"color": BENCHMARK_COLOR, "width": 2},
            )
        )
    fig.update_yaxes(type="log", title="Growth of $1")
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly returns")
    try:
        from riskmetrics.returns import monthly_returns_table

        m_tbl = monthly_returns_table(portfolio_returns)
        heat_data = m_tbl.drop(columns=["YTD"], errors="ignore")
        heat = go.Figure(
            data=go.Heatmap(
                z=heat_data.values,
                x=[str(c) for c in heat_data.columns],
                y=[str(y) for y in heat_data.index],
                colorscale=[
                    [0.0, LOSS_COLOR],
                    [0.5, "#FFFFFF"],
                    [1.0, GAIN_COLOR],
                ],
                zmid=0,
                hovertemplate="Year %{y} • Month %{x}<br>Return %{z:.2%}<extra></extra>",
            )
        )
        heat.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(heat, use_container_width=True)

        st.subheader("Annual returns")
        if "YTD" in m_tbl.columns:
            ytd = m_tbl["YTD"].dropna()
            colors = [GAIN_COLOR if v >= 0 else LOSS_COLOR for v in ytd.values]
            bar = go.Figure(
                data=go.Bar(
                    x=[str(y) for y in ytd.index],
                    y=ytd.values,
                    marker_color=colors,
                    hovertemplate="%{x}: %{y:.2%}<extra></extra>",
                )
            )
            bar.update_yaxes(tickformat=".0%")
            bar.update_layout(
                height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}
            )
            st.plotly_chart(bar, use_container_width=True)
    except Exception as exc:
        st.info(f"Monthly table unavailable: {exc}")


# ---- Risk profile tab ------------------------------------------------------
with tab_risk:
    st.subheader("Rolling 90-day annualized volatility")
    try:
        from riskmetrics.volatility import rolling_volatility

        rv = rolling_volatility(portfolio_returns, window=90, periods_per_year=252)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rv.index,
                y=rv.values,
                mode="lines",
                name="Portfolio",
                line={"color": PORTFOLIO_COLOR},
            )
        )
        if not benchmark_returns.empty and len(benchmark_returns) >= 90:
            rv_b = rolling_volatility(benchmark_returns, window=90, periods_per_year=252)
            fig.add_trace(
                go.Scatter(
                    x=rv_b.index,
                    y=rv_b.values,
                    mode="lines",
                    name=bench_ticker,
                    line={"color": BENCHMARK_COLOR},
                )
            )
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"Rolling vol unavailable: {exc}")

    st.subheader("Underwater (drawdown) curve")
    try:
        from riskmetrics.drawdown import underwater_curve

        uw = underwater_curve(portfolio_returns)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=uw.index,
                y=uw.values,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line={"color": LOSS_COLOR},
                fillcolor="rgba(213, 94, 0, 0.3)",
            )
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"Underwater curve unavailable: {exc}")

    st.subheader("Top-10 drawdown episodes")
    try:
        from riskmetrics.drawdown import drawdown_table

        dd_tbl = drawdown_table(portfolio_returns, top=10).copy()
        if not dd_tbl.empty:
            dd_tbl["drawdown"] = dd_tbl["drawdown"].map(lambda x: f"{x:.2%}")
            st.dataframe(dd_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("No underwater episodes in this window.")
    except Exception as exc:
        st.info(f"Drawdown table unavailable: {exc}")


# ---- Risk-adjusted tab -----------------------------------------------------
with tab_radj:
    st.subheader("Risk-adjusted ratios — portfolio vs benchmark")
    ratio_keys = [
        ("Sharpe", "ratios.sharpe"),
        ("Sortino", "ratios.sortino"),
        ("Calmar", "drawdown.calmar"),
        ("Sterling", "drawdown.sterling"),
        ("Information ratio", "benchmark.information_ratio"),
    ]
    grid = []
    for label, key in ratio_keys:
        grid.append(
            {
                "Metric": label,
                "Portfolio": _fmt_num(_g(ts_portfolio, key)),
                bench_ticker if not benchmark_returns.empty else "Benchmark": (
                    _fmt_num(_g(ts_benchmark, key))
                    if ts_benchmark is not None
                    else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(grid), use_container_width=True, hide_index=True)


# ---- Tail risk tab ---------------------------------------------------------
with tab_tail:
    st.subheader("VaR / CVaR table")
    arr_p = portfolio_returns.to_numpy()
    mu_p, sd_p = float(arr_p.mean()), float(arr_p.std(ddof=1))
    s_p = float(stats.skew(arr_p, bias=False))
    k_p = float(stats.kurtosis(arr_p, fisher=True, bias=False))

    def _var_param(alpha: float) -> float:
        z = stats.norm.ppf(1 - alpha)
        return float(-(mu_p + sd_p * z))

    def _var_hist(alpha: float) -> float:
        return float(-np.quantile(arr_p, 1 - alpha))

    def _var_cf(alpha: float) -> float:
        z = stats.norm.ppf(1 - alpha)
        z_cf = (
            z
            + (z**2 - 1) * s_p / 6
            + (z**3 - 3 * z) * k_p / 24
            - (2 * z**3 - 5 * z) * s_p**2 / 36
        )
        return float(-(mu_p + sd_p * z_cf))

    def _cvar_hist(alpha: float) -> float:
        cutoff = np.quantile(arr_p, 1 - alpha)
        tail = arr_p[arr_p <= cutoff]
        return float(-tail.mean()) if tail.size else float("nan")

    var_df = pd.DataFrame(
        {
            "Method": ["Historical", "Parametric (normal)", "Cornish-Fisher"],
            "VaR 95%": [_fmt_pct(_var_hist(0.95)), _fmt_pct(_var_param(0.95)), _fmt_pct(_var_cf(0.95))],
            "VaR 99%": [_fmt_pct(_var_hist(0.99)), _fmt_pct(_var_param(0.99)), _fmt_pct(_var_cf(0.99))],
            "CVaR 95%": [_fmt_pct(_cvar_hist(0.95)), "—", "—"],
            "CVaR 99%": [_fmt_pct(_cvar_hist(0.99)), "—", "—"],
        }
    )
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.subheader("Return distribution vs normal overlay")
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=arr_p,
            histnorm="probability density",
            name="Empirical",
            marker_color=PORTFOLIO_COLOR,
            opacity=0.7,
            nbinsx=60,
        )
    )
    xs = np.linspace(arr_p.min(), arr_p.max(), 400)
    hist_fig.add_trace(
        go.Scatter(
            x=xs,
            y=stats.norm.pdf(xs, mu_p, sd_p),
            mode="lines",
            name="Normal fit",
            line={"color": BENCHMARK_COLOR, "width": 2},
        )
    )
    hist_fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Q-Q plot vs normal")
    (osm, osr), (slope, intercept, _r) = stats.probplot(arr_p, dist="norm")
    qq_fig = go.Figure()
    qq_fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Sample",
            marker={"color": PORTFOLIO_COLOR, "size": 5},
        )
    )
    qq_fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode="lines",
            name="Normal reference",
            line={"color": BENCHMARK_COLOR, "dash": "dash"},
        )
    )
    qq_fig.update_xaxes(title="Theoretical quantiles")
    qq_fig.update_yaxes(title="Sample quantiles")
    qq_fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(qq_fig, use_container_width=True)


# ---- Benchmark tab ---------------------------------------------------------
with tab_bench:
    if benchmark_returns.empty:
        st.info("No benchmark data — benchmark analytics unavailable.")
    else:
        st.subheader("Alpha / Beta / R² / TE / IR / Capture")
        cards = st.columns(4)
        card_metrics = [
            ("Alpha (ann.)", "benchmark.alpha_annualized", _fmt_pct),
            ("Beta", "benchmark.beta", _fmt_num),
            ("R²", "benchmark.r_squared", _fmt_num),
            ("Tracking error", "benchmark.tracking_error", _fmt_pct),
            ("Information ratio", "benchmark.information_ratio", _fmt_num),
            ("Up-capture", "benchmark.up_capture", _fmt_num),
            ("Down-capture", "benchmark.down_capture", _fmt_num),
            ("Correlation", "benchmark.correlation", _fmt_num),
        ]
        for i, (label, key, fmt) in enumerate(card_metrics):
            cards[i % 4].metric(label, fmt(_g(ts_portfolio, key)))

        st.subheader("Rolling 90-day β")
        try:
            joined = pd.concat(
                [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
                axis=1,
                join="inner",
            ).dropna()
            cov = joined["p"].rolling(90).cov(joined["b"])
            var_b = joined["b"].rolling(90).var()
            roll_beta = (cov / var_b).dropna()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=roll_beta.index,
                    y=roll_beta.values,
                    mode="lines",
                    name="Rolling β",
                    line={"color": PORTFOLIO_COLOR},
                )
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#888888")
            fig.update_layout(
                height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Rolling β unavailable: {exc}")


# ---- Diagnostics tab -------------------------------------------------------
with tab_diag:
    st.subheader("Data coverage")
    diag_rows = []
    for t in all_tickers:
        if t in rets.columns:
            col = rets[t]
            diag_rows.append(
                {
                    "Ticker": t,
                    "Coverage %": f"{(1 - col.isna().mean()) * 100:.1f}%",
                    "Missing rows": int(col.isna().sum()),
                    "First date": str(col.dropna().index.min().date())
                    if not col.dropna().empty
                    else "—",
                    "Last date": str(col.dropna().index.max().date())
                    if not col.dropna().empty
                    else "—",
                }
            )
        else:
            diag_rows.append(
                {
                    "Ticker": t,
                    "Coverage %": "0.0%",
                    "Missing rows": "—",
                    "First date": "—",
                    "Last date": "—",
                }
            )
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    st.subheader("Run metadata")
    meta = {
        "Run timestamp (UTC)": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "Library version": f"riskmetrics v{__version__}",
        "Window": f"{start_s} → {end_s}",
        "Observations": str(n_obs),
        "Risk-free source": "FRED DGS3MO" if use_fred_rf else "0%",
    }
    st.dataframe(
        pd.DataFrame({"Field": list(meta.keys()), "Value": list(meta.values())}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Full tearsheet")
    st.dataframe(
        ts_portfolio.to_frame(name="Portfolio"), use_container_width=True
    )
