"""Random forest models, purged walk-forward backtest, metrics and signals.

Input tables are "long" format: one row per (date, ticker) with columns
'date', 'ticker', 'target_date', 'pos', the feature columns, and the targets
'future_close', 'future_return', 'target'. A single-ticker run is just a long
table with one ticker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    roc_auc_score,
)

BACKTEST_TREES = 200
FINAL_TREES = 400
MIN_TRAIN_ROWS = 750


def make_models(n_estimators: int, seed: int = 42):
    kw = dict(n_estimators=n_estimators, min_samples_leaf=30, max_features="sqrt",
              random_state=seed, n_jobs=-1)
    return RandomForestClassifier(**kw), RandomForestRegressor(**kw)


def purged_train_set(labeled: pd.DataFrame, test_date: pd.Timestamp) -> pd.DataFrame:
    """LEAKAGE FIX: keep only rows whose 6-month outcome was already known on test_date.

    A row dated D has target_date = D + HORIZON trading days. If target_date is
    after test_date, its label uses prices from after the test date, so training
    on it would leak the future. For one ticker this is the same as
    "row position <= test position - HORIZON".
    """
    test_date = pd.Timestamp(test_date)
    mask = (labeled["date"] < test_date) & (labeled["target_date"] <= test_date)
    return labeled.loc[mask]


def prob_up(clf, X) -> np.ndarray:
    classes = list(clf.classes_)
    proba = clf.predict_proba(X)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def monthly_test_points(labeled: pd.DataFrame, min_train_years: float) -> list[pd.DataFrame]:
    """For each calendar month after the warm-up, the last labeled row per ticker."""
    first_date = labeled["date"].min()
    first_test = first_date + pd.DateOffset(months=int(round(min_train_years * 12)))
    cand = labeled.loc[labeled["date"] >= first_test].copy()
    if cand.empty:
        return []
    cand["month"] = cand["date"].dt.to_period("M")
    last_idx = cand.groupby(["month", "ticker"])["date"].idxmax()
    pts = cand.loc[last_idx.values]
    return [g.drop(columns="month") for _, g in pts.groupby("month", sort=True)]


def walk_forward(labeled: pd.DataFrame, features: list[str], min_train_years: float = 3,
                 n_estimators: int = BACKTEST_TREES, min_train_rows: int = MIN_TRAIN_ROWS,
                 progress=None) -> pd.DataFrame:
    """Monthly purged walk-forward. One model fit per month predicts every ticker's
    month-end row. Training rows must have target_date <= the EARLIEST test date
    that month (so no ticker's test overlaps any training label)."""
    months = monthly_test_points(labeled, min_train_years)
    out = []
    for i, test in enumerate(months):
        if progress:
            progress(i + 1, len(months))
        cutoff = test["date"].min()
        train = purged_train_set(labeled, cutoff)
        if len(train) < min_train_rows or train["target"].nunique() < 2:
            continue
        clf, reg = make_models(n_estimators)
        clf.fit(train[features], train["target"])
        reg.fit(train[features], train["future_return"])
        p = prob_up(clf, test[features])
        r = reg.predict(test[features])
        for (_, row), pi, ri in zip(test.iterrows(), p, r):
            out.append({
                "date": row["date"], "ticker": row["ticker"], "pos": int(row["pos"]),
                "train_rows": len(train), "train_last_target_date": train["target_date"].max(),
                "current_price": float(row["Close"]),
                "actual_future_price": float(row["future_close"]),
                "actual_future_return": float(row["future_return"]),
                "actual_direction": int(row["target"]),
                "prob_up": float(pi), "prediction": int(pi >= 0.5),
                "predicted_return": float(ri),
                "predicted_price": float(row["Close"] * (1 + ri)),
            })
    return pd.DataFrame(out)


def non_overlapping(res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep predictions at least `horizon` TRADING days apart (uses 'pos').

    (The original script counted monthly backtest rows instead of trading days,
    which spaced trades ~10 years apart; this counts real trading days.)"""
    keep, last = [], None
    for idx, row in res.sort_values("date").iterrows():
        if last is None or row["pos"] - last >= horizon:
            keep.append(idx)
            last = row["pos"]
    return res.loc[keep]


def scorecard(res: pd.DataFrame, horizon: int) -> dict:
    """Metrics for one ticker's backtest rows."""
    if res is None or res.empty:
        return {}
    y, pred, p = res["actual_direction"], res["prediction"], res["prob_up"]
    base_rate = float(y.mean())  # accuracy of always guessing "up"
    acc = float(accuracy_score(y, pred))
    m = {
        "n_predictions": int(len(res)),
        "first_test": res["date"].min(), "last_test": res["date"].max(),
        "accuracy": acc,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "base_rate_always_up": base_rate,
        "majority_class_rate": max(base_rate, 1 - base_rate),
        "beats_base_rate": acc > base_rate,
        "pct_predicted_up": float(pred.mean()),
        "mae": float(mean_absolute_error(res["actual_future_price"], res["predicted_price"])),
        "rmse": float(np.sqrt(mean_squared_error(res["actual_future_price"], res["predicted_price"]))),
        "avg_pct_error": float((np.abs(res["predicted_price"] - res["actual_future_price"])
                                / res["actual_future_price"]).mean()),
    }
    no = non_overlapping(res, horizon)
    m["n_non_overlap_trades"] = int(len(no))
    if len(no):
        m["model_signal_return"] = float((1 + no["actual_future_return"] * no["prediction"]).prod() - 1)
        m["buy_hold_return"] = float((1 + no["actual_future_return"]).prod() - 1)
        m["beats_buy_hold"] = m["model_signal_return"] > m["buy_hold_return"]
    return m


def signal_for(p: float) -> str:
    if p >= 0.65:
        return "STRONG BULLISH"
    if p >= 0.55:
        return "BULLISH"
    if p <= 0.35:
        return "STRONG BEARISH"
    if p <= 0.45:
        return "BEARISH"
    return "NEUTRAL"


def fit_final(labeled: pd.DataFrame, features: list[str], n_estimators: int = FINAL_TREES):
    """Train on ALL labeled history (every label is already known today)."""
    clf, reg = make_models(n_estimators)
    clf.fit(labeled[features], labeled["target"])
    reg.fit(labeled[features], labeled["future_return"])
    return clf, reg


def forecast(clf, reg, latest: pd.DataFrame, features: list[str]) -> dict:
    p = float(prob_up(clf, latest[features])[0])
    r = float(reg.predict(latest[features])[0])
    price = float(latest["Close"].iloc[0])
    return {
        "as_of": latest.index[0] if not isinstance(latest.index, pd.RangeIndex) else latest["date"].iloc[0],
        "current_price": price, "prob_up": p, "expected_return": r,
        "predicted_price": price * (1 + r), "signal": signal_for(p),
        "confidence": abs(p - 0.5) * 2,
    }


def importance(clf, features: list[str]) -> pd.Series:
    return pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
