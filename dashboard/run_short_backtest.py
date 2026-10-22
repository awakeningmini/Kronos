"""Short Kronos vs random-forest walk-forward on real Yahoo data.

The horizon is chosen from CPU timing *before any score is computed*, using the
rule in `choose_horizon`. Accuracy is never used to pick the setup.

From the dashboard folder:

    .venv\\Scripts\\python run_short_backtest.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd

DASH = Path(__file__).resolve().parent
ROOT = DASH.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH))

import compare as C  # noqa: E402
import data as D  # noqa: E402
import features as F  # noqa: E402
import pipeline as P  # noqa: E402
import prepare as PR  # noqa: E402
from kronos_adapter import forecast_latest, load_predictor  # noqa: E402

RESULTS = DASH / "results" / "short_backtest.json"

# Frozen before looking at any error. Timing uses TSLA only.
TIMING_RULE = (
    "One Kronos-mini sample_count=1 forecast is timed on TSLA (lookback 256) "
    "before any accuracy is computed. "
    "If the 21-day forecast takes more than 25s, use horizon 10, lookback 128, 4 origins. "
    "Else time 63 days: if that takes more than 40s, use horizon 21 and 6 origins. "
    "Else time 126 days: if that takes more than 60s, use horizon 63 and 4 origins; "
    "otherwise use horizon 126 and 3 origins. step is always equal to horizon."
)


def _probe(predictor, kprices, lookback, pred_len) -> float:
    origin = kprices.index[-(pred_len + 1)]
    hist, future = PR.context_and_future(kprices, origin, lookback, pred_len)
    t0 = time.perf_counter()
    PR.call_predictor(
        predictor, hist, future.index, pred_len,
        T=1.0, top_k=0, top_p=0.9, sample_count=1,
    )
    return time.perf_counter() - t0


def choose_horizon(predictor, kprices) -> dict:
    """Return the backtest setup. Does not read predicted prices."""
    probes = {}
    t21 = _probe(predictor, kprices, 256, 21)
    probes["pred_len_21_lookback_256"] = t21
    print(f"probe 21 trading days @ lookback 256: {t21:.2f}s", flush=True)
    if t21 > 25:
        choice = {"horizon": 10, "lookback": 128, "max_origins": 4}
    else:
        t63 = _probe(predictor, kprices, 256, 63)
        probes["pred_len_63_lookback_256"] = t63
        print(f"probe 63 trading days @ lookback 256: {t63:.2f}s", flush=True)
        if t63 > 40:
            choice = {"horizon": 21, "lookback": 256, "max_origins": 6}
        else:
            t126 = _probe(predictor, kprices, 256, 126)
            probes["pred_len_126_lookback_256"] = t126
            print(f"probe 126 trading days @ lookback 256: {t126:.2f}s", flush=True)
            if t126 > 60:
                choice = {"horizon": 63, "lookback": 256, "max_origins": 4}
            else:
                choice = {"horizon": 126, "lookback": 256, "max_origins": 3}
    choice["step"] = choice["horizon"]
    choice["probes_seconds"] = probes
    choice["rule"] = TIMING_RULE
    print(
        f"locked setup: horizon={choice['horizon']} lookback={choice['lookback']} "
        f"max_origins={choice['max_origins']} step={choice['step']}",
        flush=True,
    )
    return choice


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _download_market(start: str):
    market = {}
    warnings = []
    for key, sym in D.MARKET_SYMBOLS.items():
        try:
            print(f"downloading {sym}", flush=True)
            market[key] = D.download_close(sym, start)
        except Exception as exc:
            warnings.append(f"{sym} unavailable ({exc})")
    return F.market_feature_frame(market), warnings


def main():
    ap = argparse.ArgumentParser(description="Kronos vs random-forest walk-forward on TSLA and AAPL.")
    ap.add_argument(
        "--max-origins", type=int, default=None,
        help="Override the timing rule's origin count. Horizon and lookback still come from the rule.",
    )
    ap.add_argument("--output", type=Path, default=RESULTS)
    args = ap.parse_args()

    tickers = ["TSLA", "AAPL"]
    start = "2012-01-01"
    model_key = "mini"
    print("loading Kronos-mini on CPU", flush=True)
    t_load = time.perf_counter()
    predictor = load_predictor(model_key, device="cpu")
    load_seconds = time.perf_counter() - t_load
    print(f"model load {load_seconds:.1f}s", flush=True)

    print("downloading TSLA for the timing probe", flush=True)
    tsla = D.download_prices("TSLA", start)
    setup = choose_horizon(predictor, PR.prices_to_kronos(tsla))
    if args.max_origins is not None:
        setup["max_origins"] = int(args.max_origins)
        setup["max_origins_note"] = "Override of the timing rule's origin count. Horizon was not changed."

    market_feats, warnings = _download_market(start)
    per_ticker = {}
    frames_k = []
    frames_r = []
    for ticker in tickers:
        print(f"=== {ticker} backtest ===", flush=True)
        prices = tsla if ticker == "TSLA" else D.download_prices(ticker, start)
        sector_feats = None
        sector_note = None
        try:
            etf, why = P.resolve_sector(ticker, {}, D.lookup_sector_etf)
            sector_note = {"etf": etf, "why": why}
            if etf:
                sector_feats = F.sector_feature_frame(D.download_close(etf, start))
        except Exception as exc:
            warnings.append(f"{ticker} sector lookup failed ({exc})")
        result = C.run_ticker(
            ticker, prices, predictor,
            horizon=setup["horizon"],
            lookback=setup["lookback"],
            step=setup["step"],
            max_origins=setup["max_origins"],
            market_feats=market_feats,
            sector_feats=sector_feats,
            n_estimators=100,
            min_train_rows=750,
            min_train_years=3.0,
            sample_count=1,
            T=1.0,
            top_p=0.9,
            top_k=0,
            seed=42,
            use_patterns=False,
        )
        if len(result["kronos_rows"]):
            frames_k.append(result["kronos_rows"])
        if len(result["rf_rows"]):
            frames_r.append(result["rf_rows"])
        per_ticker[ticker] = {
            "sector": sector_note,
            "n_price_rows": result["n_price_rows"],
            "price_start": result["price_start"],
            "price_end": result["price_end"],
            "n_features": result["n_features"],
            "origins_requested": result["origins_requested"],
            "rf_skipped": result["rf_skipped"],
            "seconds_per_forecast": result["seconds_per_forecast"],
            "summary": result["summary"],
            "kronos_rows": result["kronos_rows"],
            "rf_rows": result["rf_rows"],
        }
        s = result["summary"]
        print(f"{ticker} paired origins: {s.get('n_paired')}", flush=True)
        if s.get("n_paired"):
            print(
                f"  Kronos MAE {s['kronos']['mae']:.4f}  acc {s['kronos']['direction_accuracy']:.3f}",
                flush=True,
            )
            print(
                f"  RF     MAE {s['random_forest']['mae']:.4f}  "
                f"clf acc {s['random_forest']['direction_accuracy']:.3f}",
                flush=True,
            )

    combined = {}
    if frames_k and frames_r:
        combined = C.summarize_pair(pd.concat(frames_k, ignore_index=True), pd.concat(frames_r, ignore_index=True))

    print("timing a latest-bar forecast and a 4-path band on AAPL (not scored)", flush=True)
    aapl = D.download_prices("AAPL", start)
    k_aapl = PR.prices_to_kronos(aapl)
    t0 = time.perf_counter()
    one = forecast_latest(
        predictor, k_aapl, lookback=setup["lookback"], pred_len=setup["horizon"],
        n_paths=1, T=1.0, top_p=0.9, seed=7,
    )
    one_path_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    band = forecast_latest(
        predictor, k_aapl, lookback=setup["lookback"], pred_len=setup["horizon"],
        n_paths=4, T=1.0, top_p=0.9, seed=100,
    )
    band_seconds = time.perf_counter() - t0

    import torch
    payload = {
        "disclaimer": "Personal research only. Not financial advice. Not a claim that either model will make money.",
        "limitation": (
            "Kronos is used as published. It is not walk-forward retrained. Its pretraining "
            "corpus may include historical bars of these same stocks, so a backtest that ends "
            "before the pretraining cutoff is not a clean out-of-sample test. The random forest "
            "is refit at every origin with a purge gap of one horizon. Kronos only sees that "
            "ticker's OHLCV. The random forest also sees SPY, QQQ, VIX, the 10-year yield, and "
            "the sector ETF when those downloads succeed. sample_count=1, so each Kronos number "
            "is one stochastic draw (seed 42 + origin index), not an average of many paths."
        ),
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "model": "mini",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "model_load_seconds": load_seconds,
        "setup": setup,
        "start": start,
        "tickers": tickers,
        "rf_trees": 100,
        "rf_min_train_rows": 750,
        "rf_min_train_years": 3.0,
        "market_context": True,
        "sample_count": 1,
        "seed": 42,
        "warnings": warnings,
        "per_ticker": per_ticker,
        "combined": combined,
        "speed": {
            "note": "Measured on this machine, not on the owner's Windows PC.",
            "aapl_latest_one_path_seconds": one_path_seconds,
            "aapl_latest_one_path_inner_seconds": one["seconds"],
            "aapl_latest_four_path_band_seconds": band_seconds,
            "aapl_band_as_of": band["as_of"],
            "aapl_band_terminal_mean": band["terminal_mean"],
            "aapl_band_terminal_p10": band["terminal_p10"],
            "aapl_band_terminal_p90": band["terminal_p90"],
            "aapl_band_up_rate": band["up_rate"],
            "aapl_band_current_price": band["current_price"],
            "aapl_one_path_terminal": one["terminal_mean"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
