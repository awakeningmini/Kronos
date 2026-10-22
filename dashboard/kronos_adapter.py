"""Load upstream Kronos and draw forecast paths. The Kronos package is not modified.

`KronosPredictor.predict` averages `sample_count` samples internally and returns
one OHLC path, so a percentile band cannot come from a single call. Bands are
built by calling that public method once per path with `sample_count=1` and a
different torch seed, then taking percentiles of `close`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import call_predictor, future_business_days, repair_ohlc

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "mini": {
        "label": "Kronos-mini (4.1M, default)",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "label": "Kronos-small (24.7M)",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "label": "Kronos-base (102.3M)",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

_CACHE: dict[str, object] = {}


def model_spec(model_key: str) -> dict:
    key = (model_key or "mini").lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(MODELS)}.")
    return MODELS[key]


def load_predictor(model_key: str = "mini", device: str = "cpu"):
    """Load tokenizer + model from Hugging Face and pin the device.

    The upstream predictor would pick CUDA when it is visible. This dashboard
    is for a CPU-only PC, so the device is passed in (default `cpu`).
    """
    spec = model_spec(model_key)
    device = device or "cpu"
    cache_key = f"{spec['model_id']}|{device}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Importing the repo's `model` package must win over any local name.
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tokenizer = KronosTokenizer.from_pretrained(spec["tokenizer_id"])
    model = Kronos.from_pretrained(spec["model_id"])
    predictor = KronosPredictor(
        model, tokenizer, device=device, max_context=spec["max_context"]
    )
    predictor.model.eval()
    predictor.tokenizer.eval()
    _CACHE[cache_key] = predictor
    return predictor


def _seed(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))


def predict_once(
    predictor,
    hist: pd.DataFrame,
    future_index,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_count: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """One upstream predict call. Returns `(frame indexed by future dates, seconds)`."""
    _seed(seed)
    t0 = time.perf_counter()
    pred = call_predictor(
        predictor,
        hist,
        future_index,
        pred_len=len(pd.DatetimeIndex(future_index)),
        T=T,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
    )
    elapsed = time.perf_counter() - t0
    pred = pred.copy()
    pred.index = pd.DatetimeIndex(pd.to_datetime(future_index))
    return pred, elapsed


def _mean_ohlc(paths: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = np.stack([p[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=float) for p in paths], axis=0)
    mean = stacked.mean(axis=0)
    out = pd.DataFrame(mean, columns=["open", "high", "low", "close", "volume", "amount"], index=paths[0].index)
    # Averaging OHLC field-by-field can break high/low. Repair for the chart only.
    # `close` is left as the average of the sampled closes.
    close = out["close"].copy()
    out = repair_ohlc(out)
    out["close"] = close
    out["volume"] = out["volume"].clip(lower=0.0)
    out["amount"] = out["amount"].clip(lower=0.0)
    return out


def predict_paths(
    predictor,
    hist: pd.DataFrame,
    future_index,
    n_paths: int = 4,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int = 42,
) -> dict:
    """Several one-sample forecasts. The band is the 10th/90th percentile of close."""
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    paths = []
    seconds = []
    for i in range(int(n_paths)):
        pred, elapsed = predict_once(
            predictor, hist, future_index, T=T, top_k=top_k, top_p=top_p,
            sample_count=1, seed=seed + i,
        )
        paths.append(pred)
        seconds.append(elapsed)
    closes = np.stack([p["close"].to_numpy(dtype=float) for p in paths], axis=0)
    mean = _mean_ohlc(paths)
    band = pd.DataFrame(
        {
            "close_mean": closes.mean(axis=0),
            "close_p10": np.quantile(closes, 0.10, axis=0),
            "close_p50": np.quantile(closes, 0.50, axis=0),
            "close_p90": np.quantile(closes, 0.90, axis=0),
        },
        index=mean.index,
    )
    current = float(hist["close"].iloc[-1])
    terminal = closes[:, -1]
    up_rate = float(np.mean(terminal > current))
    return {
        "mean": mean,
        "band": band,
        "current_price": current,
        "terminal_mean": float(terminal.mean()),
        "terminal_p10": float(np.quantile(terminal, 0.10)),
        "terminal_p90": float(np.quantile(terminal, 0.90)),
        "expected_return": float(terminal.mean() / current - 1.0),
        "up_rate": up_rate,
        "n_paths": int(n_paths),
        "seconds": float(sum(seconds)),
        "seconds_per_path": [float(s) for s in seconds],
        "as_of": pd.Timestamp(hist.index[-1]),
    }


def forecast_latest(
    predictor,
    kronos_prices: pd.DataFrame,
    lookback: int,
    pred_len: int,
    n_paths: int = 4,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int = 42,
) -> dict:
    """Forecast the next `pred_len` weekdays from the last `lookback` bars."""
    prices = kronos_prices.sort_index()
    if len(prices) < lookback:
        raise ValueError(f"Need {lookback} bars; this series has {len(prices)}.")
    hist = prices.iloc[-lookback:]
    future_index = future_business_days(hist.index[-1], pred_len)
    out = predict_paths(
        predictor, hist, future_index, n_paths=n_paths, T=T, top_k=top_k, top_p=top_p, seed=seed
    )
    out["lookback"] = int(lookback)
    out["pred_len"] = int(pred_len)
    return out
