"""Candlestick pattern scanner (inspired by github.com/eminsk/yfinance-ta-patterns).

Works on the dashboard's own OHLC frame (columns Open/High/Low/Close). It does
NOT download anything.

Engine: TA-Lib's CDL* functions when the `TA-Lib` package is installed, else a
built-in pure-pandas detector for the same 18 patterns. Both engines only look
at the current bar and earlier bars, so a pattern flagged on date D uses no
data after D (safe to use as a model feature).

Signal convention per pattern column: +1 bullish, -1 bearish, 0 none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional fast C engine
    import talib  # type: ignore
    HAVE_TALIB = True
except Exception:  # pragma: no cover - depends on the PC
    talib = None
    HAVE_TALIB = False

# name -> (TA-Lib function, kind). kind: "bull"/"bear" fixed direction, "both" = sign decides,
# "neutral" = indecision (doji), reported as 0 direction but still listed.
PATTERNS = {
    "Doji": ("CDLDOJI", "neutral"),
    "Hammer": ("CDLHAMMER", "bull"),
    "Inverted hammer": ("CDLINVERTEDHAMMER", "bull"),
    "Hanging man": ("CDLHANGINGMAN", "bear"),
    "Shooting star": ("CDLSHOOTINGSTAR", "bear"),
    "Engulfing": ("CDLENGULFING", "both"),
    "Harami": ("CDLHARAMI", "both"),
    "Piercing line": ("CDLPIERCING", "bull"),
    "Dark cloud cover": ("CDLDARKCLOUDCOVER", "bear"),
    "Morning star": ("CDLMORNINGSTAR", "bull"),
    "Evening star": ("CDLEVENINGSTAR", "bear"),
    "Morning doji star": ("CDLMORNINGDOJISTAR", "bull"),
    "Evening doji star": ("CDLEVENINGDOJISTAR", "bear"),
    "Three white soldiers": ("CDL3WHITESOLDIERS", "bull"),
    "Three black crows": ("CDL3BLACKCROWS", "bear"),
    "Marubozu": ("CDLMARUBOZU", "both"),
    "Spinning top": ("CDLSPINNINGTOP", "neutral"),
    "Dragonfly doji": ("CDLDRAGONFLYDOJI", "bull"),
    "Gravestone doji": ("CDLGRAVESTONEDOJI", "bear"),
}

TREND_WIN = 10          # bars used to judge the prior trend for hammer/hanging man etc.
CONFLUENCE_WIN = 3      # patterns within this many bars that agree add up


def _sign_from_talib(name: str, raw: np.ndarray) -> np.ndarray:
    kind = PATTERNS[name][1]
    s = np.sign(raw).astype(int)
    if kind == "neutral":
        return np.where(s != 0, 0, 0), (s != 0)
    if kind == "bull":
        return np.where(s != 0, 1, 0), (s != 0)
    if kind == "bear":
        return np.where(s != 0, -1, 0), (s != 0)
    return s, (s != 0)


# ----------------------------------------------------------------------------
# Pure-pandas engine (simplified textbook rules; close to, not identical to, TA-Lib)
# ----------------------------------------------------------------------------
def _pandas_engine(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = body.rolling(10, min_periods=5).mean().shift(1)  # past bars only
    bull, bear = c > o, c < o
    long_body = body > avg_body
    small_body = body < 0.5 * avg_body
    doji = body <= 0.1 * rng
    mid = (o + c) / 2
    trend = c.shift(1) - c.shift(TREND_WIN + 1)  # prior trend, ends at yesterday
    down, up = trend < 0, trend > 0
    o1, c1, h1, l1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    body1, body2 = body.shift(1), body.shift(2)
    bull1, bear1, bull2, bear2 = bull.shift(1, fill_value=False), bear.shift(1, fill_value=False), \
        bull.shift(2, fill_value=False), bear.shift(2, fill_value=False)
    long1 = long_body.shift(1, fill_value=False)
    long2 = long_body.shift(2, fill_value=False)
    small1 = small_body.shift(1, fill_value=False)
    doji1 = doji.shift(1, fill_value=False)
    hammer_shape = (lower >= 2 * body) & (upper <= 0.3 * body.clip(lower=1e-12) + 0.1 * rng) & ~doji
    inv_shape = (upper >= 2 * body) & (lower <= 0.1 * rng) & ~doji

    out = {}
    out["Doji"] = (doji, 0)
    out["Hammer"] = (hammer_shape & down, 1)
    out["Hanging man"] = (hammer_shape & up, -1)
    out["Inverted hammer"] = (inv_shape & down, 1)
    out["Shooting star"] = (inv_shape & up & (o > c1), -1)
    eng_bull = bull & bear1 & (c >= o1) & (o <= c1) & (body > body1)
    eng_bear = bear & bull1 & (o >= c1) & (c <= o1) & (body > body1)
    out["Engulfing"] = (eng_bull, 1, eng_bear, -1)
    har_bull = long1 & bear1 & bull & (o > c1) & (c < o1) & (body < body1)
    har_bear = long1 & bull1 & bear & (o < c1) & (c > o1) & (body < body1)
    out["Harami"] = (har_bull, 1, har_bear, -1)
    out["Piercing line"] = (long1 & bear1 & bull & (o < l1) & (c > (o1 + c1) / 2) & (c < o1), 1)
    out["Dark cloud cover"] = (long1 & bull1 & bear & (o > h1) & (c < (o1 + c1) / 2) & (c > o1), -1)
    ms_core = long2 & bear2 & (np.maximum(o1, c1) < c2) & bull & (c > (o2 + c2) / 2)
    es_core = long2 & bull2 & (np.minimum(o1, c1) > c2) & bear & (c < (o2 + c2) / 2)
    out["Morning star"] = (ms_core & small1 & ~doji1, 1)
    out["Evening star"] = (es_core & small1 & ~doji1, -1)
    out["Morning doji star"] = (ms_core & doji1, 1)
    out["Evening doji star"] = (es_core & doji1, -1)
    tws = bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o > o1) & (o < c1) & (o1 > o2) & (o1 < c2) \
        & (upper < 0.3 * body) & long_body
    tbc = bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o < o1) & (o > c1) & (o1 < o2) & (o1 > c2) \
        & (lower < 0.3 * body) & long_body
    out["Three white soldiers"] = (tws, 1)
    out["Three black crows"] = (tbc, -1)
    maru = long_body & (upper <= 0.05 * rng) & (lower <= 0.05 * rng)
    out["Marubozu"] = (maru & bull, 1, maru & bear, -1)
    out["Spinning top"] = (~doji & (body < 0.3 * rng) & (upper > body) & (lower > body), 0)
    out["Dragonfly doji"] = (doji & (upper <= 0.1 * rng) & (lower >= 0.6 * rng), 1)
    out["Gravestone doji"] = (doji & (lower <= 0.1 * rng) & (upper >= 0.6 * rng), -1)
    return out


def detect(df: pd.DataFrame, engine: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (direction frame, hit frame, engine used).

    direction: one column per pattern, +1/-1/0 (neutral patterns are 0).
    hit: one bool column per pattern, True when the pattern fired (incl. neutral ones).
    """
    use_talib = HAVE_TALIB and engine in ("auto", "talib")
    dirs, hits = {}, {}
    if use_talib:
        o, h, l, c = (df[k].astype(float).to_numpy() for k in ("Open", "High", "Low", "Close"))
        for name, (fn, _) in PATTERNS.items():
            d, hit = _sign_from_talib(name, getattr(talib, fn)(o, h, l, c))
            dirs[name], hits[name] = d, hit
        used = "TA-Lib"
    else:
        for name, spec in _pandas_engine(df).items():
            d = np.zeros(len(df), dtype=int)
            hit = np.zeros(len(df), dtype=bool)
            for mask, sgn in zip(spec[0::2], spec[1::2]):
                m = mask.fillna(False).to_numpy(dtype=bool)
                hit |= m
                d = np.where(m, sgn, d)
            dirs[name], hits[name] = d, hit
        used = "built-in (pandas)"
    return (pd.DataFrame(dirs, index=df.index), pd.DataFrame(hits, index=df.index), used)


def confluence(dirs: pd.DataFrame, window: int = CONFLUENCE_WIN) -> pd.DataFrame:
    """Per-day bull/bear counts and a confluence score.

    bull_n / bear_n: directional patterns firing that day.
    bull_conf / bear_conf: how many directional patterns fired in the last `window`
    bars (incl. today) in that direction. score = bull_conf - bear_conf.
    Only past/current bars are used.
    """
    bull_n = (dirs > 0).sum(axis=1)
    bear_n = (dirs < 0).sum(axis=1)
    out = pd.DataFrame({"bull_n": bull_n, "bear_n": bear_n}, index=dirs.index)
    out["bull_conf"] = bull_n.rolling(window, min_periods=1).sum()
    out["bear_conf"] = bear_n.rolling(window, min_periods=1).sum()
    out["score"] = out["bull_conf"] - out["bear_conf"]
    return out


PATTERN_FEATURES = ["pat_bull_5", "pat_bear_5", "pat_net_5", "pat_bull_20", "pat_bear_20",
                    "pat_conf_score"]


def pattern_features(prices: pd.DataFrame, engine: str = "auto") -> pd.DataFrame:
    """Model features built from patterns. Row D uses bars <= D only."""
    dirs, _, _ = detect(prices, engine)
    conf = confluence(dirs)
    f = pd.DataFrame(index=prices.index)
    for w in (5, 20):
        f[f"pat_bull_{w}"] = conf["bull_n"].rolling(w, min_periods=1).sum()
        f[f"pat_bear_{w}"] = conf["bear_n"].rolling(w, min_periods=1).sum()
    f["pat_net_5"] = f["pat_bull_5"] - f["pat_bear_5"]
    f["pat_conf_score"] = conf["score"]
    return f[PATTERN_FEATURES].astype(float)


def hits_table(prices: pd.DataFrame, engine: str = "auto", last_n_days: int = 60) -> pd.DataFrame:
    """One row per pattern hit in the last `last_n_days` bars, newest first."""
    dirs, hits, _ = detect(prices, engine)
    conf = confluence(dirs)
    rows = []
    recent = prices.index[-last_n_days:]
    for d in recent:
        for name in hits.columns:
            if hits.at[d, name]:
                s = int(dirs.at[d, name])
                rows.append({"Date": d.date(), "Pattern": name,
                             "Direction": {1: "Bullish", -1: "Bearish", 0: "Neutral"}[s],
                             "Close": float(prices.at[d, "Close"]),
                             "Bull patterns (3 bars)": int(conf.at[d, "bull_conf"]),
                             "Bear patterns (3 bars)": int(conf.at[d, "bear_conf"]),
                             "Confluence score": int(conf.at[d, "score"])})
    out = pd.DataFrame(rows)
    return out.sort_values(["Date", "Pattern"], ascending=[False, True]).reset_index(drop=True) if len(out) else out


def forward_stats(prices: pd.DataFrame, engine: str = "auto", horizons=(5, 10, 20)) -> pd.DataFrame:
    """'Does this pattern work?' Average forward return after each pattern vs all days.

    Forward return h = Close[t+h] / Close[t] - 1 (entry at the signal day's close).
    'Edge' = pattern avg minus baseline avg. 'Right way %' = share of hits where the
    move matched the pattern's direction (neutral patterns: share up).
    """
    dirs, hits, _ = detect(prices, engine)
    close = prices["Close"].astype(float)
    fwd = {h: close.shift(-h) / close - 1 for h in horizons}
    rows = [{"Pattern": "All days (baseline)", "Direction": "-",
             **{f"n {h}d": int(fwd[h].notna().sum()) for h in horizons},
             **{f"Avg {h}d": fwd[h].mean() for h in horizons},
             **{f"Edge {h}d": 0.0 for h in horizons},
             **{f"Right way {h}d": (fwd[h] > 0).mean() if fwd[h].notna().any() else np.nan for h in horizons}}]
    for name in hits.columns:
        for sgn, label in ((1, "Bullish"), (-1, "Bearish"), (0, "Neutral")):
            m = hits[name] & (dirs[name] == sgn)
            if not m.any():
                continue
            row = {"Pattern": name, "Direction": label}
            for h in horizons:
                r = fwd[h][m].dropna()
                row[f"n {h}d"] = int(len(r))
                row[f"Avg {h}d"] = r.mean() if len(r) else np.nan
                row[f"Edge {h}d"] = (r.mean() - fwd[h].mean()) if len(r) else np.nan
                want = r > 0 if sgn >= 0 else r < 0
                row[f"Right way {h}d"] = want.mean() if len(r) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
