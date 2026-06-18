"""Drawdown family: drawdown series, drawdown table with peak-to-recovery durations, Ulcer/Pain indices, Calmar/Sterling ratios. drawdown_table correctly handles open (unrecovered) drawdowns with NaT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskmetrics._constants import PERIODS_PER_YEAR
from riskmetrics._typing import ReturnsLike
from riskmetrics._validation import ensure_series, validate_min_obs  # noqa: F401


def drawdown_series(returns: ReturnsLike) -> pd.Series:
    """Compute the drawdown at each timestamp as wealth-relative-to-running-peak minus one.

    The wealth curve is built as the cumulative product of ``(1 + returns)``.
    At every bar the drawdown is ``wealth / running_peak - 1``; for any
    non-pathological input every value lies in ``[-1, 0]`` (zero at a new
    high, negative when underwater).

    Args:
        returns: Per-period simple returns as a pandas Series, NumPy array,
            or list of floats. Must be finite and non-empty.

    Returns:
        Series of drawdown values aligned to the input index.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.
        TypeError: If ``returns`` is not a Series, ndarray, or list.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.10, -0.20, 0.05])
        >>> drawdown_series(r).round(4).tolist()
        [0.0, -0.2, -0.16]
    """
    r = ensure_series(returns)
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def max_drawdown(returns: ReturnsLike) -> float:
    """Return the worst (most negative) drawdown observed over the sample.

    Args:
        returns: Per-period simple returns.

    Returns:
        Single float in ``[-1, 0]``; ``0.0`` when the wealth curve never
        falls below its running peak.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.10, -0.20, 0.05])
        >>> round(max_drawdown(r), 4)
        -0.2
    """
    return float(drawdown_series(returns).min())


def drawdown_table(returns: ReturnsLike, top: int = 10) -> pd.DataFrame:
    """Enumerate the largest drawdown episodes with peak-to-recovery durations.

    A drawdown episode is a contiguous run of bars whose drawdown is strictly
    negative. The episode begins at the bar where wealth first dips below the
    prior running peak; the high-water mark itself (``peak_date``) is the bar
    where wealth last equalled that peak. The episode ends at the first bar
    where wealth recovers to (or above) the prior peak. If the wealth curve
    never recovers within the sample, ``recovery_date`` is ``pd.NaT``,
    ``is_open`` is ``True``, and ``duration_days`` is computed peak-to-last-bar
    (a censored figure).

    Boundary case: if the very first bar is already negative, there is no
    prior high-water mark inside the sample; we treat the bar *before* the
    series start as the implicit peak, and report ``peak_date`` as the first
    index value (so ``duration_days`` is measured from the start of the
    sample). This matches industry convention and avoids dropping the
    leading drawdown.

    Args:
        returns: Per-period simple returns.
        top: Maximum number of episodes to return, ranked by depth
            (deepest first). Defaults to 10.

    Returns:
        DataFrame with columns ``peak_date`` (Timestamp), ``valley_date``
        (Timestamp), ``recovery_date`` (Timestamp or NaT), ``drawdown``
        (float in ``[-1, 0]``), ``duration_days`` (int, calendar days from
        peak to recovery or to the last bar if still open), and ``is_open``
        (bool). Sorted by ``drawdown`` ascending.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values, or
            if ``top`` is not a positive integer.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2024-01-01", periods=6, freq="D")
        >>> r = pd.Series([0.10, -0.20, -0.05, 0.30, -0.10, 0.15], index=idx)
        >>> tbl = drawdown_table(r, top=2)
        >>> tbl[["peak_date", "valley_date", "recovery_date"]].iloc[0].tolist()
        [Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-03 00:00:00'), Timestamp('2024-01-06 00:00:00')]
    """
    if not isinstance(top, int) or top <= 0:
        raise ValueError(f"top must be a positive integer, got {top!r}")

    dd = drawdown_series(returns)
    n = len(dd)
    dd_values = dd.to_numpy()
    index = dd.index
    # ``duration_days`` is a calendar-day span only when the index carries
    # dates. For a plain integer/positional index (as in the documented
    # ``sterling_ratio`` example) there are no calendar days, so the duration
    # is the number of bars between the peak and the recovery (or last) bar.
    is_datetime_index = isinstance(index, pd.DatetimeIndex)

    underwater = dd_values < 0
    episodes: list[dict] = []
    i = 0
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        # Episode start: first underwater bar of a contiguous run.
        start = i
        j = i
        while j < n and underwater[j]:
            j += 1
        # j is now the recovery bar index, or n if unrecovered.
        episode_slice = slice(start, j)
        episode_dd = dd_values[episode_slice]
        valley_offset = int(np.argmin(episode_dd))
        valley_idx = start + valley_offset
        drawdown_value = float(episode_dd[valley_offset])

        # peak_date: the last bar at or before the episode where wealth equalled
        # the running peak. If start == 0 the episode opens on the first bar of
        # the sample with no prior peak inside the index; we report the first
        # index value (peak == wealth on bar 0 by construction unless r[0] < 0,
        # in which case the implicit peak is pre-sample and we still anchor at
        # index[0]).
        peak_pos = 0 if start == 0 else start - 1
        peak_date = index[peak_pos]

        if j < n:
            recovery_date = index[j]
            is_open = False
            end_pos = j
        else:
            recovery_date = pd.NaT
            is_open = True
            end_pos = n - 1

        if is_datetime_index:
            duration_days = int((index[end_pos] - peak_date).days)
        else:
            duration_days = int(end_pos - peak_pos)

        episodes.append(
            {
                "peak_date": peak_date,
                "valley_date": index[valley_idx],
                "recovery_date": recovery_date,
                "drawdown": drawdown_value,
                "duration_days": duration_days,
                "is_open": is_open,
            }
        )
        i = j + 1 if j < n else n

    columns = [
        "peak_date",
        "valley_date",
        "recovery_date",
        "drawdown",
        "duration_days",
        "is_open",
    ]
    if not episodes:
        return pd.DataFrame(
            {
                "peak_date": pd.Series([], dtype="datetime64[ns]"),
                "valley_date": pd.Series([], dtype="datetime64[ns]"),
                "recovery_date": pd.Series([], dtype="datetime64[ns]"),
                "drawdown": pd.Series([], dtype=float),
                "duration_days": pd.Series([], dtype=int),
                "is_open": pd.Series([], dtype=bool),
            }
        )[columns]

    table = pd.DataFrame(episodes, columns=columns)
    table = table.sort_values("drawdown", ascending=True, kind="stable")
    table = table.head(top).reset_index(drop=True)
    return table


def ulcer_index(returns: ReturnsLike) -> float:
    """Compute the Ulcer Index: the root-mean-square of the drawdown series.

    Defined by Martin & McCann (1989) as ``sqrt(mean(drawdown ** 2))``. The
    denominator is the total number of observations, not the count of
    underwater bars, which is the correct RMS form; some libraries implement
    it as ``sqrt(sum(dd ** 2)) / N``, which is dimensionally wrong by a
    factor of ``sqrt(N)``.

    Args:
        returns: Per-period simple returns.

    Returns:
        Non-negative float in ``[0, 1]`` for well-behaved inputs.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.10, -0.20, 0.05])
        >>> round(ulcer_index(r), 4)
        0.1479
    """
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean()))


def pain_index(returns: ReturnsLike) -> float:
    """Compute the Pain Index: the mean absolute drawdown.

    The L1 cousin of the Ulcer Index: ``mean(|drawdown|)``. Penalises
    sustained underwater periods linearly rather than quadratically.

    Args:
        returns: Per-period simple returns.

    Returns:
        Non-negative float in ``[0, 1]`` for well-behaved inputs.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.10, -0.20, 0.05])
        >>> round(pain_index(r), 4)
        0.12
    """
    return float(drawdown_series(returns).abs().mean())


def calmar_ratio(
    returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Compute the Calmar ratio: annualised CAGR divided by the absolute max drawdown.

    The CAGR is computed inline from the cumulative return and sample length
    (``(1 + r).prod() ** (periods_per_year / len(r)) - 1``) to keep the
    drawdown module self-contained. Returns ``+inf`` when the realised max
    drawdown is exactly zero (no underwater period observed) and ``nan`` when
    the CAGR itself is non-finite (e.g., wealth went non-positive).

    Args:
        returns: Per-period simple returns.
        periods_per_year: Annualisation factor matching the return frequency.
            Defaults to :data:`PERIODS_PER_YEAR` (252 trading days).

    Returns:
        Float Calmar ratio.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> isinstance(calmar_ratio(r), float)
        True
    """
    r = ensure_series(returns)
    n = len(r)
    cagr = float((1 + r).prod() ** (periods_per_year / n) - 1)
    if not np.isfinite(cagr):
        return float("nan")
    mdd = max_drawdown(r)
    if mdd == 0.0:
        return float("inf")
    return cagr / abs(mdd)


def sterling_ratio(
    returns: ReturnsLike,
    periods_per_year: int = PERIODS_PER_YEAR,
    n: int = 10,
) -> float:
    """Compute the Sterling ratio: CAGR divided by the absolute mean of the worst ``n`` drawdowns.

    Uses :func:`drawdown_table` with ``top=n`` and averages the resulting
    ``drawdown`` column. Returns ``+inf`` when no underwater episode exists
    in the sample and ``nan`` when CAGR is non-finite.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Annualisation factor matching the return frequency.
            Defaults to :data:`PERIODS_PER_YEAR`.
        n: Number of worst drawdowns to average. Defaults to 10.

    Returns:
        Float Sterling ratio.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values, or
            ``n`` is not a positive integer.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.03])
        >>> isinstance(sterling_ratio(r, n=3), float)
        True
    """
    r = ensure_series(returns)
    length = len(r)
    cagr = float((1 + r).prod() ** (periods_per_year / length) - 1)
    if not np.isfinite(cagr):
        return float("nan")
    table = drawdown_table(r, top=n)
    if table.empty:
        return float("inf")
    mean_dd = float(table["drawdown"].mean())
    if mean_dd == 0.0:
        return float("inf")
    return cagr / abs(mean_dd)


def max_drawdown_duration(returns: ReturnsLike) -> int:
    """Return the longest peak-to-recovery duration in calendar days.

    Uses :func:`drawdown_table` to enumerate every episode (no ``top`` cap)
    and reports the maximum ``duration_days``. For drawdowns still open at
    the end of the sample, the figure is censored: it is measured peak to
    last observation rather than peak to recovery, which can only
    underestimate the eventual duration.

    Args:
        returns: Per-period simple returns.

    Returns:
        Non-negative integer number of calendar days. ``0`` when the wealth
        curve never goes underwater.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2024-01-01", periods=5, freq="D")
        >>> r = pd.Series([0.10, -0.20, 0.05, 0.10, 0.05], index=idx)
        >>> max_drawdown_duration(r)
        4
    """
    r = ensure_series(returns)
    table = drawdown_table(r, top=len(r) + 1)
    if table.empty:
        return 0
    return int(table["duration_days"].max())


def underwater_curve(returns: ReturnsLike) -> pd.Series:
    """Alias for :func:`drawdown_series`, kept for dashboard ergonomics.

    Equivalent to :func:`drawdown_series`; provided because the underwater
    plot (wealth-to-peak ratio minus one over time) is one of the most
    common visualisations in performance reporting and consumers expect a
    function of this name.

    Args:
        returns: Per-period simple returns.

    Returns:
        Drawdown Series aligned to the input index.

    Raises:
        ValueError: If ``returns`` is empty or contains NaN/Inf values.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.05, -0.10, 0.02])
        >>> (underwater_curve(r) == drawdown_series(r)).all()
        True
    """
    return drawdown_series(returns)


# ``validate_min_obs`` is imported per the module contract; no metric here
# imposes an additional minimum-observation floor beyond ensure_series's
# non-empty check, but the symbol is kept available for downstream callers.
