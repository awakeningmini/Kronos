import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_prices(n=1500, seed=0, start="2012-01-02", drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def prices():
    return make_prices()
