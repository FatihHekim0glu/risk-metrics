"""Risk and performance metrics for financial return series."""

from __future__ import annotations

__version__ = "0.1.0"

from riskmetrics.benchmark import (
    CAPMResult,
    alpha,
    alpha_beta,
    beta,
    correlation,
    down_capture,
    information_ratio,
    rolling_alpha,
    rolling_beta,
    tracking_error,
    up_capture,
)
from riskmetrics.drawdown import (
    calmar_ratio,
    drawdown_series,
    drawdown_table,
    max_drawdown,
    max_drawdown_duration,
    pain_index,
    sterling_ratio,
    ulcer_index,
    underwater_curve,
)
from riskmetrics.ratios import (
    m_squared,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)
from riskmetrics.returns import (
    annualized_return_arithmetic,
    best_period,
    best_year,
    cagr,
    cumulative_returns,
    hit_rate,
    log_returns,
    monthly_returns_table,
    simple_returns,
    total_return,
    worst_period,
    worst_year,
)
from riskmetrics.tail import (
    conditional_value_at_risk,
    cornish_fisher_var,
    deflated_sharpe_ratio,
    excess_kurtosis,
    jarque_bera,
    probabilistic_sharpe_ratio,
    skewness,
    tail_ratio,
    value_at_risk,
)
from riskmetrics.tearsheet import tearsheet
from riskmetrics.volatility import (
    autocorrelation,
    downside_deviation,
    ljung_box_test,
    mean_absolute_deviation,
    realized_volatility,
    rolling_volatility,
    semi_deviation,
)

__all__ = [
    "__version__",
    # returns
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
    # volatility
    "realized_volatility",
    "downside_deviation",
    "semi_deviation",
    "mean_absolute_deviation",
    "rolling_volatility",
    "autocorrelation",
    "ljung_box_test",
    # drawdown
    "drawdown_series",
    "max_drawdown",
    "drawdown_table",
    "ulcer_index",
    "pain_index",
    "calmar_ratio",
    "sterling_ratio",
    "max_drawdown_duration",
    "underwater_curve",
    # ratios
    "sharpe_ratio",
    "sortino_ratio",
    "omega_ratio",
    "treynor_ratio",
    "m_squared",
    # tail
    "value_at_risk",
    "cornish_fisher_var",
    "conditional_value_at_risk",
    "skewness",
    "excess_kurtosis",
    "jarque_bera",
    "tail_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    # benchmark
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
    # tearsheet
    "tearsheet",
]
