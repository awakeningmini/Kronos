"""Feature engineering. Every feature for date D uses only data dated <= D.

All features are scale-free (returns, ratios, volatilities), so rows from
different tickers can be pooled into one training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 60, 120]

BASE_FEATURES = (
    ["ret_1"]
    + [f"ret_{w}" for w in WINDOWS]
    + [f"ma_ratio_{w}" for w in WINDOWS]
    + [f"vol_{w}" for w in WINDOWS]
    + ["volume_change_5", "volume_change_20", "hl_range", "close_position"]
)

TARGET_COLS = ["future_close", "future_return", "target", "target_date"]


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Christian's original 20 features (same formulas as the script)."""
    df = prices.copy()
    close = df["Close"]
    df["ret_1"] = close.pct_change()
    for w in WINDOWS:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"ma_ratio_{w}"] = close / close.rolling(w).mean()
        df[f"vol_{w}"] = df["ret_1"].rolling(w).std()

    vol = df["Volume"].astype(float)
    if (vol <= 0).mean() > 0.5:
        # Indexes (e.g. ^GSPC) have no volume -> neutral 0 instead of dropping every row
        df["volume_change_5"] = 0.0
        df["volume_change_20"] = 0.0
    else:
        vol = vol.replace(0, np.nan)
        df["volume_change_5"] = vol.pct_change(5, fill_method=None)
        df["volume_change_20"] = vol.pct_change(20, fill_method=None)

    df["hl_range"] = (df["High"] - df["Low"]) / close
    day_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_position"] = (close - df["Low"]) / day_range
    feat = BASE_FEATURES
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """future_close = Close `horizon` trading days later; target_date = that day's date."""
    df = df.copy()
    df["future_close"] = df["Close"].shift(-horizon)
    df["future_return"] = df["future_close"] / df["Close"] - 1
    df["target"] = np.where(
        df["future_close"].notna(), (df["future_close"] > df["Close"]).astype(float), np.nan
    )
    df["target_date"] = pd.Series(df.index, index=df.index).shift(-horizon)
    return df


def align_to_dates(other: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex):
    """Put `other` onto `dates` using FORWARD-FILL ONLY (never backfill).

    The value used for date D is the most recent value dated <= D, so nothing
    from the future can leak in. Dates before `other` starts stay NaN.
    """
    union = other.index.union(dates)
    return other.reindex(union).sort_index().ffill().reindex(dates)


def _series_feats(close: pd.Series, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for w in (20, 60, 120):
        out[f"{prefix}_ret_{w}"] = close.pct_change(w, fill_method=None)
    for w in (50, 200):
        out[f"{prefix}_ma_ratio_{w}"] = close / close.rolling(w).mean()
    out[f"{prefix}_vol_20"] = close.pct_change(fill_method=None).rolling(20).std()
    return out


def market_feature_frame(market: dict[str, pd.Series]) -> pd.DataFrame:
    """Build market-context features, each on its own trading calendar.

    `market` maps "spy", "qqq", "vix", "tnx" -> Close series (any may be missing).
    Returns a date-indexed frame; align it to a stock with align_to_dates().
    """
    parts = []
    for key in ("spy", "qqq"):
        if key in market and market[key] is not None and len(market[key]):
            parts.append(_series_feats(market[key].dropna(), key))
    if "vix" in market and market["vix"] is not None and len(market["vix"]):
        v = market["vix"].dropna()
        f = pd.DataFrame(index=v.index)
        f["vix_level"] = v
        f["vix_chg_5"] = v.pct_change(5)
        f["vix_chg_20"] = v.pct_change(20)
        f["vix_ma_ratio_50"] = v / v.rolling(50).mean()
        parts.append(f)
    if "tnx" in market and market["tnx"] is not None and len(market["tnx"]):
        t = market["tnx"].dropna()
        f = pd.DataFrame(index=t.index)
        f["tnx_level"] = t
        f["tnx_chg_20"] = t.diff(20)
        f["tnx_chg_60"] = t.diff(60)
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    # Outer-join on dates, then forward-fill (holiday mismatches between e.g. VIX and SPY)
    out = pd.concat(parts, axis=1).sort_index().ffill()
    return out.replace([np.inf, -np.inf], np.nan)


def sector_feature_frame(sector_close: pd.Series | None) -> pd.DataFrame:
    if sector_close is None or len(sector_close) == 0:
        return pd.DataFrame()
    return _series_feats(sector_close.dropna(), "sector").replace([np.inf, -np.inf], np.nan)


def build_ticker_frame(
    prices: pd.DataFrame,
    horizon: int,
    ticker: str = "",
    market_feats: pd.DataFrame | None = None,
    sector_feats: pd.DataFrame | None = None,
    extra: list[pd.DataFrame] | None = None,
    use_patterns: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with features + targets, feature column names)."""
    df = add_price_features(prices)
    features = list(BASE_FEATURES)

    if use_patterns:
        # Candlestick pattern counts; row D only uses candles dated <= D (no leakage).
        import patterns as PT
        pf = PT.pattern_features(prices)
        df = df.join(pf)
        features += list(pf.columns)

    if market_feats is not None and not market_feats.empty:
        m = align_to_dates(market_feats, df.index)
        df = df.join(m)
        features += list(m.columns)
        if "spy_ret_20" in m.columns:
            for w in (20, 60, 120):
                df[f"rs_spy_{w}"] = df[f"ret_{w}"] - df[f"spy_ret_{w}"]
                features.append(f"rs_spy_{w}")

    if sector_feats is not None and not sector_feats.empty:
        s = align_to_dates(sector_feats, df.index)
        # Before the sector ETF existed (XLRE 2015, XLC 2018) fall back to SPY's numbers
        for c in s.columns:
            spy_c = c.replace("sector_", "spy_")
            if spy_c in df.columns:
                s[c] = s[c].fillna(df[spy_c])
        df = df.join(s)
        features += list(s.columns)
        for w in (20, 60, 120):
            if f"sector_ret_{w}" in df.columns:
                df[f"rs_sector_{w}"] = df[f"ret_{w}"] - df[f"sector_ret_{w}"]
                features.append(f"rs_sector_{w}")

    for ex in extra or []:
        if ex is None or ex.empty:
            continue
        e = align_to_dates(ex, df.index)
        e = e[[c for c in e.columns if c not in df.columns]]
        df = df.join(e)
        features += list(e.columns)

    df = add_target(df, horizon)
    df["ticker"] = ticker
    df["pos"] = np.arange(len(df))  # trading-day position within this ticker
    return df, features


def labeled_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows with all features AND a known future outcome (usable for training/backtest)."""
    lab = frame.dropna(subset=features + ["future_close", "future_return", "target"]).copy()
    lab["target"] = lab["target"].astype(int)
    return lab


def latest_row(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Most recent row with valid features (future NOT required) for today's forecast."""
    ok = frame.dropna(subset=features)
    return ok.iloc[[-1]] if len(ok) else ok


def pool_frames(frames: dict[str, pd.DataFrame], feature_lists: dict[str, list[str]]):
    """Stack several tickers' labeled rows into one long table.

    Only features every ticker has are kept (per-ticker CSV columns are dropped
    in pooled mode). No ticker ID is used as a feature, so the model is
    ticker-agnostic. Index is a plain RangeIndex; 'date' and 'ticker' are columns.
    """
    common = None
    for t, f in feature_lists.items():
        common = list(f) if common is None else [c for c in common if c in f]
    common = common or []
    rows = []
    for t, fr in frames.items():
        lab = labeled_rows(fr, common)
        lab = lab.rename_axis("date").reset_index()
        rows.append(lab)
    pooled = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not pooled.empty:
        pooled = pooled.sort_values(["date", "ticker"]).reset_index(drop=True)
    return pooled, common
