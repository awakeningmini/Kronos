"""End-to-end pipeline: download -> features -> purged walk-forward -> final forecast.

Used by app.py (with Streamlit caching) and runnable by itself for a quick
command-line check:

    python pipeline.py TSLA AAPL --pooled --market
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pandas as pd

import data as D
import features as F
import rf_model as M


@dataclass
class Config:
    tickers: list[str]
    start: str = "2010-01-01"
    horizon: int = 126
    min_train_years: float = 3.0
    pooled: bool = False
    use_market: bool = False
    sector_overrides: dict = field(default_factory=dict)  # ticker -> "Auto" | "None" | ETF
    global_csv: pd.DataFrame | None = None
    ticker_csvs: dict = field(default_factory=dict)       # ticker -> DataFrame
    backtest_trees: int = M.BACKTEST_TREES
    final_trees: int = M.FINAL_TREES
    use_patterns: bool = False  # add candlestick-pattern features to the model


def resolve_sector(ticker, overrides, sector_lookup):
    ov = (overrides or {}).get(ticker, "Auto") or "Auto"
    if ov == "None":
        return None, "turned off by you"
    if ov != "Auto":
        return ov, "set by you"
    etf, sector = sector_lookup(ticker)
    return etf, f"auto: Yahoo sector '{sector}'" + ("" if etf else " -> no ETF match, using SPY only")


def run(cfg: Config, get_prices=D.download_prices, get_close=D.download_close,
        sector_lookup=D.lookup_sector_etf, progress=None) -> dict:
    """Run everything. `get_prices`, `get_close`, `sector_lookup` can be swapped
    for cached versions (the app) or fakes (tests)."""
    say = progress or (lambda msg, frac=None: None)
    result = {"tickers": {}, "errors": {}, "warnings": [], "config": cfg}

    # 1. Market context (shared)
    market_feats = None
    if cfg.use_market:
        market = {}
        for key, sym in D.MARKET_SYMBOLS.items():
            try:
                say(f"Downloading market data {sym}...")
                market[key] = get_close(sym, cfg.start)
            except Exception as exc:
                result["warnings"].append(f"Market series {sym} unavailable ({exc}); skipped.")
        market_feats = F.market_feature_frame(market)

    # 2. Per-ticker frames
    frames, feats, sectors, ohlc = {}, {}, {}, {}
    for t in cfg.tickers:
        try:
            say(f"Downloading {t}...")
            prices = get_prices(t, cfg.start)
            sector_feats = None
            if cfg.use_market:
                etf, why = resolve_sector(t, cfg.sector_overrides, sector_lookup)
                sectors[t] = (etf, why)
                if etf:
                    try:
                        sector_feats = F.sector_feature_frame(get_close(etf, cfg.start))
                    except Exception as exc:
                        result["warnings"].append(f"{t}: sector ETF {etf} unavailable ({exc}); using SPY only.")
                        sectors[t] = (None, f"{etf} download failed")
            extras = []
            if cfg.global_csv is not None:
                extras.append(cfg.global_csv)
            if cfg.ticker_csvs.get(t) is not None:
                tcsv = cfg.ticker_csvs[t].add_prefix(f"{t}_".replace("^", ""))
                extras.append(tcsv)
                if cfg.pooled:
                    result["warnings"].append(
                        f"{t}: per-ticker CSV columns can't be used in pooled mode (other tickers "
                        "don't have them), so they were left out. Upload it as 'All tickers' or switch to single mode.")
            frame, flist = F.build_ticker_frame(prices, cfg.horizon, t, market_feats, sector_feats, extras,
                                               use_patterns=cfg.use_patterns)
            if len(F.labeled_rows(frame, flist)) == 0:
                raise ValueError("no rows with complete features and a known 6-month outcome "
                                 "(history too short, or an uploaded CSV starts too late).")
            frames[t], feats[t] = frame, flist
            ohlc[t] = prices[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as exc:
            result["errors"][t] = str(exc)

    if not frames:
        return result

    # 3. Backtest + final models
    def one_group(names, flist, label):
        lab_long, used = F.pool_frames({n: frames[n] for n in names}, {n: flist for n in names})
        say(f"Backtesting {label} ({len(lab_long):,} training rows)...", 0.0)
        res = M.walk_forward(lab_long, used, cfg.min_train_years, cfg.backtest_trees,
                             progress=lambda i, n: say(f"Backtesting {label}: month {i}/{n}", i / n))
        say(f"Training final model for {label}...")
        clf, reg = M.fit_final(lab_long, used, cfg.final_trees)
        imp = M.importance(clf, used)
        for n in names:
            latest = F.latest_row(frames[n], used)
            r_n = res[res["ticker"] == n].set_index("date") if not res.empty else pd.DataFrame()
            lab_n = lab_long[lab_long["ticker"] == n]
            result["tickers"][n] = {
                "forecast": M.forecast(clf, reg, latest, used) if len(latest) else None,
                "scorecard": M.scorecard(r_n.reset_index(), cfg.horizon) if len(r_n) else {},
                "backtest": r_n,
                "importance": imp,
                "features": used,
                "n_features": len(used),
                "labeled_range": (lab_n["date"].min(), lab_n["date"].max()),
                "labeled_rows": int(len(lab_n)),
                "price_range": (frames[n].index.min(), frames[n].index.max()),
                "sector": sectors.get(n),
                "trained_on": label,
                "prices": ohlc.get(n),
            }
            if not len(r_n):
                result["warnings"].append(
                    f"{n}: not enough history for a backtest (need ~{cfg.min_train_years:g} years of "
                    f"labeled data plus {M.MIN_TRAIN_ROWS} purged training rows). Try an earlier start date.")

    if cfg.pooled and len(frames) > 1:
        common = None
        for n in frames:
            common = list(feats[n]) if common is None else [c for c in common if c in feats[n]]
        one_group(list(frames), common, "pooled model (" + ", ".join(frames) + ")")
    else:
        if cfg.pooled:
            result["warnings"].append("Pooled mode needs 2+ tickers; ran single-ticker mode.")
        for n in frames:
            one_group([n], feats[n], f"{n} only")
    return result


def summary_table(result: dict) -> pd.DataFrame:
    rows = []
    for t, r in result["tickers"].items():
        f, s = r["forecast"] or {}, r["scorecard"] or {}
        rows.append({
            "Ticker": t,
            "Data as of": f.get("as_of").date() if f.get("as_of") is not None else None,
            "Price": f.get("current_price"),
            "P(higher)": f.get("prob_up"),
            "Est. price": f.get("predicted_price"),
            "Exp. return": f.get("expected_return"),
            "Signal": f.get("signal"),
            "Confidence": f.get("confidence"),
            "Backtest accuracy": s.get("accuracy"),
            "Always-up rate": s.get("base_rate_always_up"),
            "Beats base rate?": ("Yes" if s.get("beats_base_rate") else "No") if s else None,
            "ROC-AUC": s.get("roc_auc"),
            "Avg % error": s.get("avg_pct_error"),
            "Model signal return": s.get("model_signal_return"),
            "Buy & hold return": s.get("buy_hold_return"),
            "Tests": s.get("n_predictions"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the model without the dashboard.")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--min-train-years", type=float, default=3)
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--patterns", action="store_true", help="add candlestick-pattern features")
    a = ap.parse_args()
    cfg = Config([D.clean_ticker(t) for t in a.tickers], a.start, a.horizon, a.min_train_years,
                 a.pooled, a.market, use_patterns=a.patterns)
    res = run(cfg, progress=lambda m, f=None: print(m) if f is None or f in (0.0, 1.0) else None)
    for w in res["warnings"]:
        print("WARNING:", w)
    for t, e in res["errors"].items():
        print(f"ERROR {t}: {e}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary_table(res).to_string(index=False))
    for t, r in res["tickers"].items():
        print(f"\n== {t} ({r['trained_on']}, {r['n_features']} features, sector={r['sector']}) ==")
        for k, v in r["scorecard"].items():
            print(f"  {k:24s} {v}")
        print("  forecast:", r["forecast"])
        print("  top features:", r["importance"].head(8).round(4).to_dict())
    print("\nPersonal research only - not financial advice.")
