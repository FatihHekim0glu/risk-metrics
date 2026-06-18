# Design rationale

This document records the non-obvious decisions in `riskmetrics`. Each item is
deliberate, has a referenced precedent, and is unlikely to change without a
breaking-release bump and a corresponding entry in `CHANGELOG.md`.

## 1. Sortino divides by N, not N_downside

The Sortino downside deviation is computed as
`sqrt(sum(min(r - target, 0)^2) / N)`, where `N` is the total number of
observations, not the count of observations below the target. This follows
the original Sortino & Price (1994) convention and matches the implementations
in `empyrical`, `quantstats`, `pyfolio`, and `vectorbt`. Dividing by
`N_downside` produces a higher and more flattering ratio but is inconsistent
with the literature and the rest of the Python ecosystem.

## 2. `drawdown_table` separates `is_open` from `recovery_date=NaT`

When a drawdown is still underwater at the end of the sample, we return
`recovery_date=NaT` and `is_open=True` rather than silently closing it at the
last available date. Several historical libraries treat the end of the sample
as a recovery, which understates current drawdown length and inflates the
"average recovery" statistic. Users who want the legacy behaviour can post-
process the table; users who want the truth get it by default.

## 3. `statsmodels.OLS` with `add_constant` and HAC errors for alpha/beta

`alpha_beta` regresses excess returns on excess benchmark returns using
`statsmodels.OLS` with an explicit constant from `sm.add_constant`, then
reports HAC (Newey-West) standard errors. Financial return series exhibit
serial correlation and conditional heteroskedasticity, so the default OLS
standard errors are not trustworthy and the resulting alpha t-statistic is
the first thing a reviewer will check.

## 4. `smart=True` Lo-2002 adjustment is opt-in

The Lo (2002) adjustment shrinks the Sharpe ratio when returns are
autocorrelated. We default `smart=False` because the user-facing definition
of Sharpe is the simple one, and silently applying an adjustment that changes
the number would surprise people coming from textbooks, Excel, or other
libraries. Setting `smart=True` makes the choice visible in the call site
and in the docstring.

## 5. VaR uses `method="lower"`

Historical VaR is computed via `numpy.quantile(..., method="lower")` so the
result is an actual observed loss in the sample rather than a linear
interpolation between two adjacent observations. This matches the practical
interpretation a risk manager wants: "we have actually seen a loss at least
this bad". For very small samples it differs from `method="linear"` by one
order statistic, and the difference is documented in the function docstring.

## 6. Excess kurtosis (Fisher) is the default

Where we report kurtosis (e.g. as an input to `cornish_fisher_var` and in the
tearsheet), we use the Fisher definition, where a normal distribution gives 0,
rather than the Pearson definition where normal gives 3. Excess kurtosis is
easier to read at a glance and is the convention in `scipy.stats.kurtosis`
and in most finance textbooks since Tsay.

## 7. Parity benchmark is empyrical-reloaded

`empyrical-reloaded` is a maintained fork of the original `empyrical` library
(which has been abandoned since Quantopian shut down). It is the most cited
Python reference for these metrics and is what people compare against, so it
is the natural target for our Layer 3 parity tests. Where we deliberately
diverge from it (e.g. `drawdown_table` not closing open drawdowns), the
parity test is replaced with a divergence test that pins the difference.

## 8. `ffn` is excluded from parity tests

The `ffn` library has confirmed bugs in its Ulcer Index and Sortino
implementations (issues #193 and #254 on github.com/pmorissette/ffn). We do
not want parity with broken code, so `ffn` is not in the parity test matrix.
Where users have asked about specific `ffn` numbers, we explain the
divergence in the docstring rather than copy the bug.

## 9. All pairwise metrics start with `align_inner`

The most common bug in financial-metrics code is operating on two series
whose indices look the same but are not, leading to silent NaN propagation
or, worse, a metric computed on misaligned data. Every pairwise function
in this library begins with `r, b = align_inner(r, b)` so the caller does
not have to think about it. The cost is one extra inner-join per call,
which is negligible compared to the time spent acquiring the data.

## 10. Annualisation is 252 trading days

We annualise daily equity returns with a factor of `252`, not `365` or
`365.25`. This is the industry standard for US equities (252 trading days
per year on average) and matches every reference text and competing
library. For non-daily frequencies the user passes `periods_per_year`
explicitly. We document this, lock it as the default, and do not entertain
calendar-day variants in the core API.
