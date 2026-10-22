"""Data loading: Yahoo Finance downloads (via yfinance), ticker list storage,
sector ETF lookup and user CSV parsing.

NOTE: yfinance is an unofficial, community-built way to read Yahoo Finance.
It is fine for personal research but it can break or be rate-limited
whenever Yahoo changes things.
"""
from __future__ import annotations

import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "tickers.json"

DEFAULT_SETTINGS = {"tickers": ["TSLA"], "sector_overrides": {}}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

MARKET_SYMBOLS = {"spy": "SPY", "qqq": "QQQ", "vix": "^VIX", "tnx": "^TNX"}

# yfinance "sector" text -> SPDR sector ETF
SECTOR_TO_ETF = {
    "technology": "XLK",
    "consumer cyclical": "XLY",
    "consumer discretionary": "XLY",
    "financial services": "XLF",
    "financials": "XLF",
    "financial": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "health care": "XLV",
    "industrials": "XLI",
    "basic materials": "XLB",
    "materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
}
SECTOR_ETFS = ["XLK", "XLY", "XLF", "XLE", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC", "XLP"]

TICKER_RE = re.compile(r"^[A-Z0-9.\-\^=]{1,15}$")


# ----------------------------------------------------------------------------
# Ticker list (saved in tickers.json next to the app)
# ----------------------------------------------------------------------------
def clean_ticker(text: str) -> str:
    t = (text or "").strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError(
            f"'{text}' doesn't look like a ticker. Use letters/numbers like TSLA, BRK-B or ^GSPC."
        )
    return t


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tickers = [clean_ticker(t) for t in data.get("tickers", [])]
        overrides = {clean_ticker(k): v for k, v in data.get("sector_overrides", {}).items()}
        return {"tickers": tickers or ["TSLA"], "sector_overrides": overrides}
    except FileNotFoundError:
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    except Exception:
        # Corrupt file: fall back to defaults instead of crashing the app
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(settings: dict, path: Path = SETTINGS_FILE) -> None:
    Path(path).write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_prices(ticker: str, start: str) -> pd.DataFrame:
    """Daily OHLCV (split/dividend adjusted). Raises ValueError if nothing came back."""
    import yfinance as yf

    df = yf.download(ticker, start=str(start), auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(
            f"No data downloaded for {ticker}. Check the symbol, or Yahoo may be "
            "temporarily blocking/rate-limiting requests."
        )
    df = _flatten(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    df = df[REQUIRED_COLS].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"
    return df


def download_close(symbol: str, start: str) -> pd.Series:
    return download_prices(symbol, start)["Close"].rename(symbol)


def lookup_sector_etf(ticker: str) -> tuple[str | None, str]:
    """Return (sector ETF or None, sector text) using yfinance .info. Never raises."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        sector = str(info.get("sector") or "").strip()
        etf = SECTOR_TO_ETF.get(sector.lower())
        return etf, sector or "unknown"
    except Exception as exc:  # network / Yahoo changes
        return None, f"lookup failed ({type(exc).__name__})"


# ----------------------------------------------------------------------------
# User CSV
# ----------------------------------------------------------------------------
def parse_extra_csv(raw: bytes | str, prefix: str = "csv_") -> pd.DataFrame:
    """Validate an uploaded CSV: needs a Date column + at least one numeric column.

    Returns a DataFrame indexed by date (sorted, de-duplicated) whose columns are
    the numeric columns, renamed with `prefix`. Raises ValueError with a plain-
    English message when something is wrong.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        raise ValueError(f"Couldn't read the file as CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The CSV is empty.")
    date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    if date_col is None:
        raise ValueError(
            f"The CSV needs a column named 'Date'. Columns found: {list(df.columns)}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dates = pd.to_datetime(df[date_col], errors="coerce")
    bad = int(dates.isna().sum())
    if bad == len(df):
        raise ValueError("None of the values in the Date column could be read as dates (use YYYY-MM-DD).")
    df = df.loc[dates.notna()].copy()
    df.index = pd.DatetimeIndex(dates[dates.notna()]).tz_localize(None).normalize()
    df = df.drop(columns=[date_col])

    numeric = {}
    skipped = []
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() == 0:
            skipped.append(str(c))
        else:
            numeric[f"{prefix}{str(c).strip().replace(' ', '_')}"] = col
    if not numeric:
        raise ValueError(
            "The CSV has no numeric columns besides Date. Add at least one column of numbers."
        )
    out = pd.DataFrame(numeric, index=df.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "Date"
    out.attrs["skipped_columns"] = skipped
    out.attrs["bad_dates"] = bad
    return out
