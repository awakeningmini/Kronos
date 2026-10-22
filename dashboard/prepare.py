"""Turn Yahoo daily bars into Kronos inputs, and pick walk-forward origins.

Nothing in this module loads the neural net. The walk-forward split is the
leakage guard for Kronos:

* the context window ends on the origin date (that close is known that day)
* the label is the close `horizon` trading days later
* those future bars are never placed in the context

When `step` is at least `horizon`, successive label windows do not overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC_COLS = ["open", "high", "low", "close", "volume", "amount"]
PRICE_COLS = ["open", "high", "low", "close", "volume"]


def repair_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Make high the highest price on the bar and low the lowest.

    A valid Yahoo bar is unchanged. A bar with high below close, or with high
    and low swapped, is pulled back into a real candle. Close is not edited.
    """
    out = df.copy()
    prices = out[["open", "high", "low", "close"]]
    out["high"] = prices.max(axis=1)
    out["low"] = prices.min(axis=1)
    return out


def prices_to_kronos(prices: pd.DataFrame) -> pd.DataFrame:
    """Yahoo-style OHLCV (Open/High/Low/Close/Volume) -> Kronos columns.

    `amount` is not in the free Yahoo download. It is filled as
    volume * typical price (high+low+close)/3, which matches the spirit of
    the upstream predictor (it invents amount from volume and price when the
    column is missing). Rows with a missing OHLC price are dropped.
    """
    if prices is None or len(prices) == 0:
        raise ValueError("No price rows to convert.")
    df = prices.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Price frame is missing {missing}.")
    out = df[PRICE_COLS].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise ValueError("Every row had a missing open/high/low/close.")
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    out = repair_ohlc(out)
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    out["amount"] = out["volume"] * typical
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "Date"
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=OHLC_COLS)
    return out[OHLC_COLS].astype(float)


def future_business_days(last_date, n: int) -> pd.DatetimeIndex:
    """`n` weekdays after `last_date`. Exchange holidays are not removed."""
    if n < 1:
        raise ValueError("n must be at least 1")
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    return pd.bdate_range(start, periods=int(n))


def model_inputs(hist: pd.DataFrame, future_index) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Frames the upstream `KronosPredictor.predict` method expects.

    Timestamp *values* are datetimes. The Series index is a plain range so the
    predictor does not confuse the row index with the calendar.
    """
    missing = [c for c in OHLC_COLS if c not in hist.columns]
    if missing:
        raise ValueError(f"History is missing {missing}.")
    if len(hist) < 2:
        raise ValueError("Need at least 2 historical bars.")
    future_index = pd.DatetimeIndex(pd.to_datetime(future_index))
    if len(future_index) < 1:
        raise ValueError("Need at least 1 future timestamp.")
    if hist[OHLC_COLS].isnull().any().any():
        raise ValueError("History still has NaNs; drop them before predicting.")
    x = hist[OHLC_COLS].astype(float).reset_index(drop=True)
    x_ts = pd.Series(pd.to_datetime(pd.DatetimeIndex(hist.index)), name="timestamps")
    y_ts = pd.Series(pd.to_datetime(future_index), name="timestamps")
    x_ts = x_ts.reset_index(drop=True)
    y_ts = y_ts.reset_index(drop=True)
    if x_ts.dt.tz is not None:
        x_ts = x_ts.dt.tz_localize(None)
    if y_ts.dt.tz is not None:
        y_ts = y_ts.dt.tz_localize(None)
    return x, x_ts, y_ts


def assert_gap(hist_index, future_index, origin) -> None:
    """Context must end on the origin, and must not share a date with the label window."""
    hist_index = pd.DatetimeIndex(pd.to_datetime(hist_index))
    future_index = pd.DatetimeIndex(pd.to_datetime(future_index))
    origin = pd.Timestamp(origin)
    if len(hist_index) == 0 or len(future_index) == 0:
        raise AssertionError("empty context or future window")
    if pd.Timestamp(hist_index.max()) != origin:
        raise AssertionError(
            f"context ends {hist_index.max()}, not the origin {origin}"
        )
    if not (pd.Timestamp(future_index.min()) > origin):
        raise AssertionError(
            f"future starts {future_index.min()} which is not after origin {origin}"
        )
    shared = hist_index.intersection(future_index)
    if len(shared):
        raise AssertionError(f"context and future share dates, first {shared[0]}")


def select_origins(
    dates,
    lookback: int,
    horizon: int,
    step: int,
    max_origins: int | None = None,
) -> list[pd.Timestamp]:
    """Origin dates anchored on the most recent valid bar.

    Position `i` is valid when the previous `lookback` bars exist (including i)
    and the next `horizon` bars exist. Origins are spaced `step` trading days
    apart, and the newest valid position is always included when any exist.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    first_pos = lookback - 1
    last_pos = len(dates) - horizon - 1
    if last_pos < first_pos:
        return []
    positions = list(range(int(last_pos), int(first_pos) - 1, -int(step)))
    positions.reverse()
    if max_origins is not None:
        if max_origins < 1:
            raise ValueError("max_origins must be at least 1")
        positions = positions[-int(max_origins):]
    return [pd.Timestamp(dates[i]) for i in positions]


def context_and_future(
    kronos_prices: pd.DataFrame, origin, lookback: int, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one series at `origin`.

    Returns `(history, future_bars)`. `future_bars` is for scoring only.
    Callers must pass `history` to the model and only the future *dates*
    as the timestamps to forecast.
    """
    prices = kronos_prices.sort_index()
    origin = pd.Timestamp(origin)
    if origin not in prices.index:
        raise ValueError(f"{origin.date()} is not a trading day in this series.")
    loc = prices.index.get_loc(origin)
    if not isinstance(loc, (int, np.integer)):
        raise ValueError(f"Duplicate dates at {origin.date()}.")
    loc = int(loc)
    start = loc - lookback + 1
    if start < 0:
        raise ValueError(
            f"Need {lookback} bars before {origin.date()} inclusive; have {loc + 1}."
        )
    if loc + horizon >= len(prices):
        raise ValueError(
            f"Need {horizon} bars after {origin.date()}; series ends {prices.index[-1].date()}."
        )
    hist = prices.iloc[start: loc + 1]
    future = prices.iloc[loc + 1: loc + 1 + horizon]
    assert_gap(hist.index, future.index, origin)
    if len(hist) != lookback or len(future) != horizon:
        raise AssertionError("window lengths do not match lookback/horizon")
    return hist.copy(), future.copy()


def call_predictor(
    predictor,
    hist: pd.DataFrame,
    future_index,
    pred_len: int,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_count: int = 1,
) -> pd.DataFrame:
    """Call `predictor.predict` on history only. `future_index` is dates, not prices."""
    x, x_ts, y_ts = model_inputs(hist, future_index)
    if len(y_ts) != pred_len:
        raise ValueError(f"pred_len={pred_len} but {len(y_ts)} future timestamps were given")
    if x_ts.max() >= y_ts.min():
        raise ValueError("history timestamps are not strictly before the forecast dates")
    pred = predictor.predict(
        df=x,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=T,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    if not isinstance(pred, pd.DataFrame) or "close" not in pred.columns:
        raise ValueError("predictor.predict did not return a DataFrame with a close column")
    if len(pred) != pred_len:
        raise ValueError(f"predictor returned {len(pred)} rows, expected {pred_len}")
    return pred
