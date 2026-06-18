"""Coverage for drawdown helpers beyond the hand-calc suite:
table validation and the no-drawdown branch, pain/ulcer, Calmar/Sterling
infinities, duration, and the underwater alias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmetrics.drawdown import (
    calmar_ratio,
    drawdown_series,
    drawdown_table,
    max_drawdown_duration,
    pain_index,
    sterling_ratio,
    ulcer_index,
    underwater_curve,
)


def test_drawdown_table_rejects_non_positive_top() -> None:
    r = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError, match="positive integer"):
        drawdown_table(r, top=0)


def test_drawdown_table_empty_when_monotone_up() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    r = pd.Series([0.01, 0.02, 0.03, 0.01], index=idx)
    table = drawdown_table(r)
    assert table.empty
    assert list(table.columns) == [
        "peak_date",
        "valley_date",
        "recovery_date",
        "drawdown",
        "duration_days",
        "is_open",
    ]


def test_drawdown_table_recovery_and_open_flags() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    # Recovers once, then dips again and never recovers.
    r = pd.Series([0.10, -0.20, 0.30, 0.05, -0.10, -0.05], index=idx)
    table = drawdown_table(r)
    # The deepest episode recovered; check there is at least one open episode.
    assert table["is_open"].any()
    open_row = table[table["is_open"]].iloc[0]
    assert pd.isna(open_row["recovery_date"])


def test_pain_index_is_mean_absolute_drawdown() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    dd = drawdown_series(r)
    assert pain_index(r) == pytest.approx(float(dd.abs().mean()), abs=1e-12)


def test_ulcer_index_is_rms_drawdown() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    dd = drawdown_series(r)
    assert ulcer_index(r) == pytest.approx(float(np.sqrt((dd**2).mean())), abs=1e-12)


def test_calmar_infinite_without_drawdown() -> None:
    r = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert calmar_ratio(r) == float("inf")


def test_sterling_infinite_without_drawdown() -> None:
    r = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert sterling_ratio(r, n=3) == float("inf")


def test_sterling_finite_with_drawdown() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.03], index=idx)
    out = sterling_ratio(r, n=3)
    assert np.isfinite(out)


def test_max_drawdown_duration_zero_without_drawdown() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    r = pd.Series([0.01, 0.02, 0.01, 0.03], index=idx)
    assert max_drawdown_duration(r) == 0


def test_max_drawdown_duration_counts_calendar_days() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    r = pd.Series([0.10, -0.20, 0.05, 0.10, 0.05], index=idx)
    assert max_drawdown_duration(r) == 4


def test_sterling_ratio_handles_integer_index() -> None:
    # The documented example uses a plain integer index (no dates). The
    # peak-to-recovery duration then has no calendar days, so the table must
    # fall back to a bar count rather than crashing on ``.days``.
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.03])
    out = sterling_ratio(r, n=3)
    assert np.isfinite(out)


def test_max_drawdown_duration_integer_index_counts_bars() -> None:
    # Peak at bar 0; wealth never reclaims that high, so the episode stays open
    # to the last bar (position 4). Duration is the bar count 4 - 0 = 4.
    r = pd.Series([0.10, -0.20, 0.05, 0.10, 0.05])
    assert max_drawdown_duration(r) == 4


def test_drawdown_table_integer_index_open_episode() -> None:
    # Never recovers; the open-episode branch must also use the bar count.
    r = pd.Series([0.10, -0.20, -0.05])
    table = drawdown_table(r)
    assert table["is_open"].all()
    assert pd.isna(table.iloc[0]["recovery_date"])
    # Peak at position 0, last bar at position 2 -> 2 bars.
    assert table.iloc[0]["duration_days"] == 2


def test_underwater_curve_aliases_drawdown_series() -> None:
    r = pd.Series([0.05, -0.10, 0.02])
    assert (underwater_curve(r) == drawdown_series(r)).all()
