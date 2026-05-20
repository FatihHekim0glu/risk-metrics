"""Shared type aliases for public function signatures."""

from __future__ import annotations

from typing import TypeAlias, Union

import numpy as np
import pandas as pd

ReturnsLike: TypeAlias = Union[pd.Series, np.ndarray, list[float]]
PricesLike: TypeAlias = Union[pd.Series, pd.DataFrame]
