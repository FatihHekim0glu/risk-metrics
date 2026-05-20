"""Data layer for risk-metrics: price/risk-free fetching, caching, and return computation.

Provides ticker price retrieval (yfinance with Stooq fallback), FRED risk-free rates,
return computation, and trading-calendar alignment.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_INSTALL_HINT = (
    "Install with `pip install riskmetrics[data]` to enable price fetching."
)

# Trading days per year used to convert annualized rates to per-day rates.
_PERIODS_PER_YEAR = 252


def _normalize_tickers(tickers: str | list[str]) -> list[str]:
    """Coerce ticker input to a non-empty list of strings."""
    if isinstance(tickers, str):
        tickers = [tickers]
    if not tickers:
        raise ValueError("`tickers` must be a non-empty string or list of strings.")
    if not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError("All tickers must be non-empty strings.")
    return [t.strip() for t in tickers]


def _resolve_cache_dir(cache_dir: str | Path | None, source: str) -> Path:
    """Resolve the cache directory, creating it if necessary."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "riskmetrics" / source
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _require_pyarrow() -> None:
    """Ensure pyarrow is available for parquet I/O."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "pyarrow is required for parquet caching. " + _INSTALL_HINT
        ) from exc


def _to_stooq_symbol(ticker: str) -> str:
    """Stooq uses suffixes like '.US' for US equities; append if no dot present."""
    return ticker if "." in ticker else f"{ticker}.US"


def _slice_dates(
    frame: pd.DataFrame | pd.Series,
    start: str | None,
    end: str | None,
) -> pd.DataFrame | pd.Series:
    """Slice a date-indexed frame to the [start, end] window."""
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    if start_ts is not None and end_ts is not None:
        return frame.loc[start_ts:end_ts]
    if start_ts is not None:
        return frame.loc[start_ts:]
    if end_ts is not None:
        return frame.loc[:end_ts]
    return frame


def _read_cached_ticker(path: Path) -> pd.Series | None:
    """Read a per-ticker parquet cache as a Series, or return None if absent."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    series = df.iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def _write_cached_ticker(path: Path, series: pd.Series, ticker: str) -> None:
    """Persist a per-ticker price Series as parquet."""
    out = series.to_frame(name=ticker)
    out.index.name = "date"
    out.to_parquet(path)


def _fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str | None,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Pull adjusted close prices from yfinance using an impersonating session."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. " + _INSTALL_HINT) from exc
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        raise ImportError("curl_cffi is required. " + _INSTALL_HINT) from exc

    session = curl_requests.Session(impersonate="chrome")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        session=session,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex on columns when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=-1)
    else:
        # Single ticker path: yfinance returns flat columns.
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    close = close.sort_index().dropna(how="all")
    # Preserve requested ticker order where possible.
    cols = [t for t in tickers if t in close.columns]
    return close[cols] if cols else close


def _fetch_stooq(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Pull close prices from Stooq via pandas_datareader, one ticker at a time."""
    try:
        from pandas_datareader import data as pdr
    except ImportError as exc:
        raise ImportError(
            "pandas_datareader is required for the Stooq source. " + _INSTALL_HINT
        ) from exc

    frames: list[pd.Series] = []
    for ticker in tickers:
        symbol = _to_stooq_symbol(ticker)
        try:
            raw = pdr.DataReader(symbol, "stooq", start, end)
        except Exception as exc:  # pragma: no cover - network failures
            warnings.warn(f"Stooq fetch failed for {ticker!r}: {exc}", stacklevel=2)
            continue
        if raw is None or raw.empty or "Close" not in raw.columns:
            continue
        series = raw["Close"].sort_index().rename(ticker)
        series.index = pd.to_datetime(series.index)
        frames.append(series)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def get_prices(
    tickers: str | list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    source: str = "yfinance",
    cache_dir: str | Path | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices for one or more tickers with on-disk caching.

    Args:
        tickers: Single ticker string or list of tickers.
        start: ISO date string for the inclusive start of the window.
        end: ISO date string for the inclusive end of the window, or None for today.
        source: ``"yfinance"`` (default) or ``"stooq"``. yfinance failures
            (rate limits or empty responses) trigger an automatic Stooq fallback.
        cache_dir: Directory for per-ticker parquet caches. Defaults to
            ``~/.cache/riskmetrics/{source}``.
        auto_adjust: Forwarded to yfinance; the resulting ``Close`` column is
            already split/dividend-adjusted when True.

    Returns:
        Wide DataFrame indexed by date with one column per requested ticker.

    Raises:
        ValueError: If ``tickers`` is empty or ``source`` is unknown.
        ImportError: If the chosen backend's optional dependency is missing.

    Example:
        >>> px = get_prices(["AAPL", "MSFT"], start="2020-01-01")
        >>> px.tail()
    """
    if source not in {"yfinance", "stooq"}:
        raise ValueError(
            f"Unknown source {source!r}; expected 'yfinance' or 'stooq'."
        )

    ticker_list = _normalize_tickers(tickers)
    _require_pyarrow()
    cache_path = _resolve_cache_dir(cache_dir, source)

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    freshness_threshold = today - pd.tseries.offsets.BDay(1)

    fetched_frames: dict[str, pd.Series] = {}
    needs_fetch: list[str] = []
    fetch_starts: dict[str, str] = {}

    for ticker in ticker_list:
        cached = _read_cached_ticker(cache_path / f"{ticker}.parquet")
        if cached is not None and not cached.empty:
            if cached.index.max() >= freshness_threshold:
                fetched_frames[ticker] = cached
                continue
            # Incremental fetch from day after cache's last date.
            fetch_starts[ticker] = (
                cached.index.max() + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
            fetched_frames[ticker] = cached
        else:
            fetch_starts[ticker] = start
        needs_fetch.append(ticker)

    if needs_fetch:
        # Group tickers by their effective fetch start to minimize round trips.
        groups: dict[str, list[str]] = {}
        for ticker in needs_fetch:
            groups.setdefault(fetch_starts[ticker], []).append(ticker)

        for group_start, group_tickers in groups.items():
            new_data = pd.DataFrame()
            if source == "yfinance":
                try:
                    new_data = _fetch_yfinance(
                        group_tickers, group_start, end, auto_adjust
                    )
                except Exception as exc:
                    # Catch yfinance.exceptions.YFRateLimitError without importing it.
                    if exc.__class__.__name__ == "YFRateLimitError":
                        warnings.warn(
                            "yfinance rate-limited; falling back to Stooq.",
                            stacklevel=2,
                        )
                        new_data = _fetch_stooq(group_tickers, group_start, end)
                    else:
                        raise
                if new_data.empty:
                    warnings.warn(
                        "yfinance returned no data; falling back to Stooq.",
                        stacklevel=2,
                    )
                    new_data = _fetch_stooq(group_tickers, group_start, end)
            else:  # source == "stooq"
                new_data = _fetch_stooq(group_tickers, group_start, end)

            for ticker in group_tickers:
                if ticker not in new_data.columns:
                    continue
                new_series = new_data[ticker].dropna()
                if new_series.empty:
                    continue
                if ticker in fetched_frames:
                    combined = pd.concat([fetched_frames[ticker], new_series])
                else:
                    combined = new_series
                combined = combined.sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                fetched_frames[ticker] = combined
                _write_cached_ticker(
                    cache_path / f"{ticker}.parquet", combined, ticker
                )

    if not fetched_frames:
        return pd.DataFrame()

    wide = pd.concat(
        [fetched_frames[t].rename(t) for t in ticker_list if t in fetched_frames],
        axis=1,
    )
    wide = wide.sort_index()
    wide.index.name = "date"
    return _slice_dates(wide, start, end)


def get_risk_free(
    start: str,
    end: str | None = None,
    series: str = "DGS3MO",
    cache_dir: str | Path | None = None,
) -> pd.Series:
    """Fetch a daily per-period risk-free rate series from FRED.

    Args:
        start: ISO date string for the inclusive start of the window.
        end: ISO date string for the inclusive end of the window, or None for today.
        series: FRED series ID. ``"DGS3MO"`` (default) is the 3-month Treasury
            constant-maturity yield on investment basis. ``"DTB3"`` is the
            discount-basis 3-month bill; a warning is emitted because the
            discount-to-yield conversion is not applied here.
        cache_dir: Directory for the raw FRED parquet cache. Defaults to
            ``~/.cache/riskmetrics/fred``.

    Returns:
        Calendar-daily Series of per-day risk-free rates (e.g., 0.00019 ≈ 5% annual),
        forward-filled across non-FRED-publishing days.

    Raises:
        ImportError: If ``pandas_datareader`` is not installed.

    Example:
        >>> rf = get_risk_free("2020-01-01")
        >>> rf.head()
    """
    if series == "DTB3":
        warnings.warn(
            "DTB3 is quoted on a discount basis; this function does not apply "
            "the discount-to-yield conversion. Use DGS3MO for investment basis.",
            stacklevel=2,
        )

    _require_pyarrow()
    cache_path = _resolve_cache_dir(cache_dir, "fred")
    cache_file = cache_path / f"_fred_{series}.parquet"

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    freshness_threshold = today - pd.tseries.offsets.BDay(1)

    cached_annual: pd.Series | None = None
    if cache_file.exists():
        cached_df = pd.read_parquet(cache_file)
        if not cached_df.empty:
            cached_annual = cached_df.iloc[:, 0]
            cached_annual.index = pd.to_datetime(cached_annual.index)
            cached_annual = cached_annual.sort_index()

    needs_refresh = (
        cached_annual is None
        or cached_annual.empty
        or cached_annual.index.max() < freshness_threshold
    )

    if needs_refresh:
        try:
            from pandas_datareader import data as pdr
        except ImportError as exc:
            raise ImportError(
                "pandas_datareader is required for FRED data. " + _INSTALL_HINT
            ) from exc

        fetch_start = (
            (cached_annual.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if cached_annual is not None and not cached_annual.empty
            else start
        )
        try:
            raw = pdr.DataReader(series, "fred", fetch_start, end)
        except Exception as exc:
            if cached_annual is None or cached_annual.empty:
                raise
            warnings.warn(
                f"FRED fetch failed ({exc}); using cached values only.",
                stacklevel=2,
            )
            raw = pd.DataFrame()

        if not raw.empty:
            new_series = raw[series].dropna() / 100.0
            new_series.index = pd.to_datetime(new_series.index)
            if cached_annual is not None:
                combined = pd.concat([cached_annual, new_series])
            else:
                combined = new_series
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            cached_annual = combined
            out_df = cached_annual.to_frame(name=series)
            out_df.index.name = "date"
            out_df.to_parquet(cache_file)

    if cached_annual is None or cached_annual.empty:
        return pd.Series(dtype=float, name=series)

    annual = _slice_dates(cached_annual, start, end)
    if annual.empty:
        return pd.Series(dtype=float, name=series)

    # Convert annualized rate to per-day compounded rate, then forward-fill
    # over calendar days so weekends/holidays carry the last published rate.
    daily_rate = (1.0 + annual) ** (1.0 / _PERIODS_PER_YEAR) - 1.0
    calendar_end = pd.Timestamp(end) if end else daily_rate.index.max()
    calendar_start = pd.Timestamp(start) if start else daily_rate.index.min()
    full_idx = pd.date_range(calendar_start, calendar_end, freq="D")
    daily_rate = daily_rate.reindex(full_idx).ffill()
    daily_rate.name = series
    daily_rate.index.name = "date"
    return daily_rate


def compute_returns(
    prices: pd.DataFrame | pd.Series,
    method: str = "simple",
) -> pd.DataFrame | pd.Series:
    """Compute simple or log returns from a price series/frame.

    Args:
        prices: Price Series or DataFrame indexed by date.
        method: ``"simple"`` for arithmetic returns, ``"log"`` for log returns.

    Returns:
        Returns object of the same shape as ``prices``, with the first row
        dropped (NaN from the differencing).

    Raises:
        ValueError: If ``method`` is not one of ``"simple"`` or ``"log"``.

    Example:
        >>> rets = compute_returns(prices, method="log")
    """
    if method == "simple":
        return prices.pct_change().dropna()
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    raise ValueError(
        f"Unknown method {method!r}; expected 'simple' or 'log'."
    )


def align_to_calendar(
    returns: pd.DataFrame,
    calendar: str = "NYSE",
) -> pd.DataFrame:
    """Reindex a returns frame onto an approximate trading-day calendar.

    For mixed asset classes (e.g., crypto plus equities) this collapses 7-day
    crypto histories onto business days. The implementation uses
    ``pd.bdate_range`` (Mon-Fri) and forward-fills once; this is an
    approximation of the NYSE calendar and ignores US market holidays. For
    production use, supply a holidays-aware calendar.

    Args:
        returns: Returns DataFrame indexed by date.
        calendar: Calendar name. Only ``"NYSE"`` is recognized; other values
            emit a warning and fall back to plain business days.

    Returns:
        DataFrame reindexed to business days within the original date range,
        forward-filled once to carry non-trading-day values forward.

    Example:
        >>> aligned = align_to_calendar(returns, calendar="NYSE")
    """
    if returns.empty:
        return returns

    if calendar != "NYSE":
        warnings.warn(
            f"Calendar {calendar!r} not recognized; using plain business days. "
            "For exchange holidays, use a calendar-aware library in production.",
            stacklevel=2,
        )

    bdays = pd.bdate_range(returns.index.min(), returns.index.max())
    return returns.reindex(bdays).ffill(limit=1).dropna(how="all")
