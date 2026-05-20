"""Shared type aliases for public function signatures."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

ReturnsLike: TypeAlias = pd.Series | np.ndarray | list[float]
PricesLike: TypeAlias = pd.Series | pd.DataFrame
