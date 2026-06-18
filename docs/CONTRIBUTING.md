# Contributing to riskmetrics

Thanks for your interest in contributing. This document covers local setup, the
workflow for adding a new metric, code style, test conventions, and what a good
pull request looks like.

## Local setup

```bash
git clone https://github.com/fatihhekimoglu/risk-metrics.git
cd risk-metrics
uv sync --all-extras
uv run pytest
```

The full test suite should pass on a clean checkout. If it does not, open an
issue before changing anything. That is a real bug and we want to know about it
before any new work lands on top of it.

Useful one-off commands:

```bash
uv run pytest tests/test_sharpe.py -k "hand_calc"   # run a single test file
uv run ruff check src tests                         # lint
uv run ruff format src tests                        # format
uv run mypy src                                     # type check
```

## How to add a new metric

The recommended workflow keeps each metric small, well-tested, and visible.

1. **Implement** the metric in the module it belongs to (`returns.py`,
   `drawdowns.py`, `tail.py`, `regression.py`, etc.). Keep the public function
   accepting a `pandas.Series` or `pandas.DataFrame` of returns plus whatever
   parameters it needs. Use `riskmetrics._align.align_inner` for any pairwise
   metric.
2. **Hand-calc test** (Layer 1). Write a test in `tests/` that computes the
   metric on a tiny fixed input by hand, asserts the expected value to a
   reasonable tolerance, and includes the arithmetic in a comment. This is the
   non-negotiable layer, and every metric needs one.
3. **Empyrical parity test** (Layer 3) if the metric exists in
   `empyrical-reloaded` and we agree with its definition. Use `pytest.importorskip`
   so the test is skipped when the dependency is not installed.
4. **Tearsheet integration**. If the metric is something a user would want in a
   one-line summary of a strategy, add it to `riskmetrics.tearsheet` so it shows
   up in the standard report.
5. **Changelog**. Add a one-line entry under `## [Unreleased]` in `CHANGELOG.md`.

## Code style

- **Formatting and linting** are handled by `ruff`. Run `ruff format` before
  committing and `ruff check` should pass with no warnings.
- **Type hints** are required on every public function. We run `mypy` in lax
  mode (`--ignore-missing-imports`, no `--strict`). Annotate what you can and
  do not fight the type checker over pandas edge cases.
- **Docstrings** use the Google style with `Args`, `Returns`, `Raises`, and an
  `Example` section. The example must be runnable as a doctest.
- **Naming**. Functions use `snake_case`. Arguments named `r` are a return
  series, `prices` are price levels, `rf` is the risk-free rate, `benchmark` is
  a benchmark return series.

## Test conventions

We organise tests into four layers and every metric should be covered by at
least Layer 1.

- **Layer 1, Hand-calc.** Tiny fixed input, expected value computed by hand,
  arithmetic shown in a comment. Catches sign errors, off-by-one,
  annualisation mistakes.
- **Layer 2, Edge cases.** Empty input, single observation, all-zero returns,
  all-negative returns, NaNs in the middle of the series, misaligned indices.
- **Layer 3, Empyrical parity.** Where a metric also exists in
  `empyrical-reloaded`, assert near-equality on a real basket of returns.
  Documented divergences (e.g. our `drawdown_table` not silently closing open
  drawdowns) are pinned with comments.
- **Layer 4, Property tests.** `hypothesis`-driven invariants: Sharpe is
  scale-invariant after rescaling risk-free, max drawdown is monotone
  non-positive, etc.

Tests live in `tests/` and follow the layout `test_<module>.py`. A property
test file is named `test_<module>_properties.py`.

## Commit messages

- Imperative mood, lowercase first word: `add cornish-fisher var`,
  `fix sortino divisor`, `tighten drawdown_table docstring`.
- Subject line ≤ 72 characters.
- Body wrapped at 72 columns, explaining *why* if it is not obvious from the
  diff.
- **No co-author or generated-with trailers.** The author is the
  person who wrote the code.

## Pull request review checklist

Before requesting review, please confirm:

- [ ] `uv run ruff check src tests` is clean.
- [ ] `uv run ruff format --check src tests` is clean.
- [ ] `uv run mypy src` is clean.
- [ ] `uv run pytest` passes locally.
- [ ] Every new public function has a docstring with `Args`, `Returns`,
      `Raises`, and an `Example`.
- [ ] `CHANGELOG.md` has a one-line entry under `## [Unreleased]`.
- [ ] No unrelated changes are bundled in.
- [ ] The PR description explains the *why*, not just the *what*.
