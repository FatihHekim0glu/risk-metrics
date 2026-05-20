"""Project-wide numeric conventions.

These constants define the trading-calendar assumptions used across the
package. The industry standard of 252 trading days per year (and the derived
21-per-month, 5-per-week figures) is applied uniformly so that annualisation
and rolling-window helpers stay consistent across modules.
"""

from __future__ import annotations

PERIODS_PER_YEAR: int = 252
TRADING_DAYS_PER_MONTH: int = 21
TRADING_DAYS_PER_WEEK: int = 5
