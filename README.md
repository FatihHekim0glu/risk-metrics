# riskmetrics

Risk and performance metrics for financial return series. Validated against `empyrical-reloaded`.

[![CI](https://github.com/FatihHekim0glu/risk-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/FatihHekim0glu/risk-metrics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FatihHekim0glu/risk-metrics/blob/main/LICENSE)

## Install

```bash
pip install git+https://github.com/FatihHekim0glu/risk-metrics.git@v0.1.0
```

## Quick start

```python
import pandas as pd
from riskmetrics import sharpe_ratio, max_drawdown, drawdown_table

returns = pd.Series(
    [0.01, -0.02, 0.015, -0.005, 0.008, -0.012, 0.02, 0.003],
    index=pd.date_range("2026-01-01", periods=8, freq="B"),
)

print("Sharpe ratio:", sharpe_ratio(returns))
print("Max drawdown:", max_drawdown(returns))
print(drawdown_table(returns))
```

## What's included

- **returns**: return aggregation, cumulative and rolling returns, annualisation helpers.
- **volatility**: sample volatility, rolling volatility, downside deviation.
- **drawdown**: drawdown series, max drawdown, and a peak-to-recovery `drawdown_table`.
- **ratios**: Sharpe, Sortino, Calmar, information ratio.
- **tail**: Value-at-Risk (historical and parametric) and conditional VaR.
- **benchmark**: alpha, beta, tracking error, and active-return statistics.

An optional `dashboard` extra adds a Streamlit interface for interactive exploration.

## Why this library

- `drawdown_table` reports peak-to-recovery duration correctly, returning `NaT` for episodes that have not yet recovered at the end of the sample.
- `sharpe_ratio` supports the Lo (2002) autocorrelation adjustment via `smart=True`, which corrects naive annualisation when returns are serially correlated.
- Every metric is parity-tested against `empyrical-reloaded` so numerical results match a well-known reference within tight tolerances.

## Development

```bash
git clone https://github.com/FatihHekim0glu/risk-metrics.git
cd risk-metrics
uv sync --all-extras
uv run pytest
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
