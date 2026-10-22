"""Data-prep tests. No network and no Kronos weights."""
import numpy as np
import pandas as pd
import pytest

import prepare as PR
from conftest import make_prices


def test_prices_to_kronos_amount_and_columns(prices):
    out = PR.prices_to_kronos(prices)
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "amount"]
    i = 40
    typical = (out["high"].iloc[i] + out["low"].iloc[i] + out["close"].iloc[i]) / 3.0
    assert out["amount"].iloc[i] == pytest.approx(out["volume"].iloc[i] * typical)
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out.index.is_monotonic_increasing
    assert out.index.tz is None


def test_repair_swaps_inverted_high_low():
    idx = pd.bdate_range("2020-01-01", periods=2)
    raw = pd.DataFrame(
        {"open": [10.0, 11.0], "high": [9.0, 10.0], "low": [12.0, 13.0], "close": [11.0, 12.0]},
        index=idx,
    )
    raw["volume"] = 100.0
    out = PR.prices_to_kronos(raw)
    assert (out["high"] >= out["low"]).all()
    assert out["high"].iloc[0] == pytest.approx(12.0)
    assert out["low"].iloc[0] == pytest.approx(9.0)


def test_future_business_days_skip_weekends():
    # Friday 2024-01-05 -> next weekday is Monday 2024-01-08
    days = PR.future_business_days(pd.Timestamp("2024-01-05"), 3)
    assert list(days) == list(pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]))
    assert all(d.weekday() < 5 for d in days)


def test_select_origins_anchors_on_latest_and_respects_windows():
    idx = pd.bdate_range("2020-01-01", periods=100)
    origins = PR.select_origins(idx, lookback=20, horizon=10, step=10)
    pos = {pd.Timestamp(d): i for i, d in enumerate(idx)}
    assert origins[-1] == idx[100 - 10 - 1]
    assert origins[0] >= idx[19]
    for a, b in zip(origins, origins[1:]):
        assert pos[b] - pos[a] == 10
        # label window (a+1 .. a+horizon] ends exactly on the next origin when step == horizon
        assert pos[a] + 10 == pos[b]
    short = PR.select_origins(idx, lookback=20, horizon=10, step=10, max_origins=2)
    assert short == origins[-2:]
    assert PR.select_origins(idx, lookback=90, horizon=20, step=5) == []


def test_context_gap_ignores_prices_after_origin(prices):
    k = PR.prices_to_kronos(prices)
    origin = k.index[200]
    hist, future = PR.context_and_future(k, origin, lookback=40, horizon=15)
    PR.assert_gap(hist.index, future.index, origin)
    assert len(hist) == 40 and len(future) == 15
    assert hist.index.max() == origin
    assert future.index.min() > origin
    assert set(hist.index).isdisjoint(future.index)

    tampered = k.copy()
    tampered.loc[tampered.index > origin, "close"] *= 4
    hist2, future2 = PR.context_and_future(tampered, origin, 40, 15)
    pd.testing.assert_frame_equal(hist, hist2)
    assert not np.allclose(future["close"], future2["close"])


def test_model_inputs_dates_do_not_overlap(prices):
    k = PR.prices_to_kronos(prices)
    origin = k.index[180]
    hist, future = PR.context_and_future(k, origin, 30, 8)
    x, x_ts, y_ts = PR.model_inputs(hist, future.index)
    assert len(x) == 30 and len(y_ts) == 8
    assert x_ts.max() < y_ts.min()
    assert list(x.columns) == PR.OHLC_COLS
    # Passing the future *prices* is a caller bug the splitter itself does not do:
    # model_inputs only reads hist columns, and y timestamps are the index.
    assert float(x["close"].iloc[-1]) == pytest.approx(float(hist["close"].iloc[-1]))
    with pytest.raises(ValueError):
        PR.model_inputs(hist, [])
