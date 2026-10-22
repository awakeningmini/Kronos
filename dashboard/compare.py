"""Paired walk-forward: Kronos vs the dashboard random forest, same origins.

Random-forest rows use the same purge as the existing dashboard: a training
row is kept only when its outcome date is on or before the origin, so the
label gap is one full horizon.

Kronos is not retrained. Its only leakage control is the context/future split
in `prepare.context_and_future`. It may still have seen older history of these
stocks during pretraining; that limit is reported with the results, not hidden.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import features as F
import prepare as PR
import rf_model as M


def price_metrics(df: pd.DataFrame) -> dict:
    """Shared scorecard. Direction uses the `prediction` column (1 = called up)."""
    if df is None or len(df) == 0:
        return {}
    err = df["predicted_price"].astype(float) - df["actual_future_price"].astype(float)
    pct = err.abs() / df["actual_future_price"].astype(float)
    y = df["actual_direction"].astype(int)
    pred = df["prediction"].astype(int)
    return {
        "n": int(len(df)),
        "direction_accuracy": float((pred == y).mean()),
        "base_rate_always_up": float(y.mean()),
        "beats_always_up": bool((pred == y).mean() > y.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "avg_pct_error": float(pct.mean()),
        "mean_predicted_return": float(df["predicted_return"].astype(float).mean()),
        "mean_actual_return": float(df["actual_future_return"].astype(float).mean()),
    }


def naive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Predict that price does not change. Direction call is 'not up'."""
    out = df.copy()
    out["predicted_price"] = out["current_price"].astype(float)
    out["predicted_return"] = 0.0
    out["prediction"] = 0
    return out


def paired_origins(price_index, labeled_dates, lookback, horizon, step, max_origins):
    """Origins that Kronos can score and that the random forest has a label for."""
    candidates = PR.select_origins(price_index, lookback, horizon, step, max_origins=None)
    have = set(pd.Timestamp(d) for d in pd.to_datetime(labeled_dates))
    good = [d for d in candidates if pd.Timestamp(d) in have]
    if max_origins is not None:
        good = good[-int(max_origins):]
    return good


def evaluate_kronos_at_origins(
    predictor,
    kronos_prices: pd.DataFrame,
    origins,
    lookback: int,
    horizon: int,
    ticker: str,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_count: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Point forecast at each origin. `sample_count` is averaged inside one upstream call."""
    rows = []
    for i, origin in enumerate(origins):
        hist, future = PR.context_and_future(kronos_prices, origin, lookback, horizon)
        # Dates only cross this line. Prices in `future` are read after the call.
        t0 = time.perf_counter()
        try:
            import torch

            torch.manual_seed(int(seed) + i)
        except Exception:
            pass
        pred = PR.call_predictor(
            predictor, hist, future.index, horizon, T=T, top_k=top_k, top_p=top_p,
            sample_count=sample_count,
        )
        elapsed = time.perf_counter() - t0
        current = float(hist["close"].iloc[-1])
        actual = float(future["close"].iloc[-1])
        predicted = float(pred["close"].iloc[-1])
        rows.append({
            "date": pd.Timestamp(origin),
            "ticker": ticker,
            "current_price": current,
            "actual_future_price": actual,
            "actual_future_return": actual / current - 1.0,
            "actual_direction": int(actual > current),
            "predicted_price": predicted,
            "predicted_return": predicted / current - 1.0,
            "prediction": int(predicted > current),
            "seconds": float(elapsed),
            "target_date": pd.Timestamp(future.index[-1]),
        })
    return pd.DataFrame(rows)


def evaluate_rf_at_origins(
    labeled: pd.DataFrame,
    features: list[str],
    origins,
    ticker: str,
    n_estimators: int = 100,
    min_train_rows: int = M.MIN_TRAIN_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """One purged forest per origin, for a single ticker's test row that day."""
    rows = []
    skipped = []
    lab = labeled
    for origin in origins:
        origin = pd.Timestamp(origin)
        test = lab.loc[(lab["ticker"] == ticker) & (lab["date"] == origin)]
        if test.empty:
            skipped.append({"date": origin, "reason": "no labeled row"})
            continue
        train = M.purged_train_set(lab, origin)
        if len(train) < min_train_rows or train["target"].nunique() < 2:
            skipped.append({
                "date": origin,
                "reason": f"only {len(train)} purged training rows (need {min_train_rows} and both classes)",
            })
            continue
        # Gap check: no training label is allowed to look past the origin.
        if not (train["target_date"] <= origin).all():
            raise AssertionError("purged training set contains a label after the origin")
        if not (train["date"] < origin).all():
            raise AssertionError("purged training set contains the origin row")
        clf, reg = M.make_models(n_estimators, seed=seed)
        clf.fit(train[features], train["target"])
        reg.fit(train[features], train["future_return"])
        row = test.iloc[0]
        prob = float(M.prob_up(clf, test[features])[0])
        ret = float(reg.predict(test[features])[0])
        current = float(row["Close"])
        actual = float(row["future_close"])
        rows.append({
            "date": origin,
            "ticker": ticker,
            "current_price": current,
            "actual_future_price": actual,
            "actual_future_return": float(row["future_return"]),
            "actual_direction": int(row["target"]),
            "prob_up": prob,
            "prediction": int(prob >= 0.5),
            "reg_prediction": int(ret > 0),
            "predicted_return": ret,
            "predicted_price": current * (1.0 + ret),
            "train_rows": int(len(train)),
            "train_last_target_date": pd.Timestamp(train["target_date"].max()),
            "target_date": pd.Timestamp(row["target_date"]),
        })
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out


def _abs_pct(df: pd.DataFrame) -> pd.Series:
    return (df["predicted_price"] - df["actual_future_price"]).abs() / df["actual_future_price"]


def summarize_pair(kronos_df: pd.DataFrame, rf_df: pd.DataFrame) -> dict:
    """Metrics on the dates both models actually scored."""
    if kronos_df is None or rf_df is None or kronos_df.empty or rf_df.empty:
        return {"n_paired": 0}
    k = kronos_df.copy()
    r = rf_df.copy()
    k["date"] = pd.to_datetime(k["date"])
    r["date"] = pd.to_datetime(r["date"])
    keys = ["date", "ticker"] if "ticker" in k.columns and "ticker" in r.columns else ["date"]
    merged_k = k.merge(r[keys], on=keys, how="inner")
    merged_r = r.merge(k[keys], on=keys, how="inner")
    if merged_k.empty:
        return {"n_paired": 0}
    km = price_metrics(merged_k)
    rm = price_metrics(merged_r)
    # Regressor direction uses the same dates. Rebuild prediction from reg_prediction.
    reg = merged_r.copy()
    if "reg_prediction" in reg.columns:
        reg["prediction"] = reg["reg_prediction"].astype(int)
        rm["regressor_direction_accuracy"] = price_metrics(reg)["direction_accuracy"]
    naive = price_metrics(naive_frame(merged_k))
    kk = merged_k.sort_values(keys)
    rr = merged_r.sort_values(keys)
    k_err = _abs_pct(kk).to_numpy()
    r_err = _abs_pct(rr).to_numpy()
    return {
        "n_paired": int(len(merged_k)),
        "first_origin": pd.Timestamp(kk["date"].min()),
        "last_origin": pd.Timestamp(kk["date"].max()),
        "kronos": km,
        "random_forest": rm,
        "naive_flat": naive,
        "kronos_closer_count": int((k_err < r_err).sum()),
        "random_forest_closer_count": int((r_err < k_err).sum()),
        "tie_count": int((k_err == r_err).sum()),
        "kronos_mae_minus_rf_mae": float(km["mae"] - rm["mae"]),
        "kronos_avg_pct_error_minus_rf": float(km["avg_pct_error"] - rm["avg_pct_error"]),
    }


def run_ticker(
    ticker: str,
    prices: pd.DataFrame,
    predictor,
    horizon: int,
    lookback: int,
    step: int,
    max_origins: int | None,
    market_feats: pd.DataFrame | None = None,
    sector_feats: pd.DataFrame | None = None,
    extra: list | None = None,
    use_patterns: bool = False,
    n_estimators: int = 100,
    min_train_rows: int = M.MIN_TRAIN_ROWS,
    T: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    sample_count: int = 1,
    seed: int = 42,
    min_train_years: float = 3.0,
) -> dict:
    """One ticker. Random forest is fit per origin on this ticker only (not pooled)."""
    kronos_prices = PR.prices_to_kronos(prices)
    frame, feats = F.build_ticker_frame(
        prices, horizon, ticker, market_feats, sector_feats, extra, use_patterns=use_patterns
    )
    labeled, used = F.pool_frames({ticker: frame}, {ticker: feats})
    origins = paired_origins(
        kronos_prices.index, labeled["date"] if len(labeled) else [],
        lookback, horizon, step, max_origins=None,
    )
    # Same warm-up idea as the monthly dashboard backtest: do not score origins
    # that sit inside the first `min_train_years` of labeled history.
    if len(labeled) and min_train_years:
        first_ok = pd.Timestamp(labeled["date"].min()) + pd.DateOffset(
            months=int(round(float(min_train_years) * 12))
        )
        origins = [d for d in origins if pd.Timestamp(d) >= first_ok]
    if max_origins is not None:
        origins = origins[-int(max_origins):]

    k_df = evaluate_kronos_at_origins(
        predictor, kronos_prices, origins, lookback, horizon, ticker,
        T=T, top_k=top_k, top_p=top_p, sample_count=sample_count, seed=seed,
    ) if origins else pd.DataFrame()
    rf_df = evaluate_rf_at_origins(
        labeled, used, origins, ticker, n_estimators=n_estimators,
        min_train_rows=min_train_rows, seed=seed,
    ) if origins else pd.DataFrame()
    if len(k_df) and len(rf_df):
        both = k_df.merge(rf_df, on="date", suffixes=("_k", "_rf"))
        if len(both):
            if not np.allclose(both["actual_future_price_k"], both["actual_future_price_rf"], rtol=0, atol=1e-6):
                raise AssertionError("Kronos and the random forest are not scoring the same future close.")
            if not np.allclose(both["current_price_k"], both["current_price_rf"], rtol=0, atol=1e-6):
                raise AssertionError("Kronos and the random forest are not using the same origin close.")
    summary = summarize_pair(k_df, rf_df)
    seconds = k_df["seconds"].astype(float) if len(k_df) else pd.Series(dtype=float)
    return {
        "ticker": ticker,
        "n_price_rows": int(len(prices)),
        "price_start": pd.Timestamp(prices.index.min()) if len(prices) else None,
        "price_end": pd.Timestamp(prices.index.max()) if len(prices) else None,
        "n_features": int(len(used)),
        "features": list(used),
        "origins_requested": [pd.Timestamp(d) for d in origins],
        "rf_skipped": list(getattr(rf_df, "attrs", {}).get("skipped", [])),
        "kronos_rows": k_df,
        "rf_rows": rf_df,
        "summary": summary,
        "seconds_per_forecast": {
            "n": int(seconds.shape[0]),
            "mean": float(seconds.mean()) if len(seconds) else None,
            "min": float(seconds.min()) if len(seconds) else None,
            "max": float(seconds.max()) if len(seconds) else None,
        },
    }
