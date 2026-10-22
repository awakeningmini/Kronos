"""Walk-forward gap tests with a fake Kronos predictor. No network."""
import numpy as np
import pandas as pd

import compare as C
import prepare as PR
from conftest import make_prices


class RecordingPredictor:
    """Returns a flat close equal to the last context close. Records what it was shown."""

    def __init__(self):
        self.seen = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.seen.append({
            "n": len(df),
            "x_max": pd.Timestamp(pd.to_datetime(x_timestamp).max()),
            "y_min": pd.Timestamp(pd.to_datetime(y_timestamp).min()),
            "y_len": len(pd.to_datetime(y_timestamp)),
            "last_close": float(df["close"].iloc[-1]),
            "columns": list(df.columns),
        })
        last = float(df["close"].iloc[-1])
        return pd.DataFrame(
            {c: np.full(pred_len, last) for c in ["open", "high", "low", "close", "volume", "amount"]}
        )


def test_price_metrics_hand_values():
    df = pd.DataFrame({
        "predicted_price": [110.0, 90.0],
        "actual_future_price": [100.0, 100.0],
        "actual_direction": [1, 0],
        "prediction": [1, 1],
        "predicted_return": [0.1, -0.1],
        "actual_future_return": [0.0, 0.0],
        "current_price": [100.0, 100.0],
    })
    m = C.price_metrics(df)
    assert m["n"] == 2
    assert m["direction_accuracy"] == 0.5
    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert m["avg_pct_error"] == 0.1
    assert m["base_rate_always_up"] == 0.5
    assert m["beats_always_up"] is False
    naive = C.price_metrics(C.naive_frame(df))
    assert naive["mae"] == 0.0
    assert naive["mean_predicted_return"] == 0.0


def test_fake_kronos_vs_rf_has_purge_gap_and_paired_prices():
    prices = make_prices(n=420, seed=3, start="2018-01-02")
    pred = RecordingPredictor()
    out = C.run_ticker(
        "AAA", prices, pred,
        horizon=10, lookback=30, step=10, max_origins=3,
        n_estimators=8, min_train_rows=40, min_train_years=0, seed=1,
    )
    assert out["summary"]["n_paired"] == 3
    assert len(pred.seen) == 3
    k = out["kronos_rows"].sort_values("date")
    rf = out["rf_rows"].sort_values("date")
    assert list(k["date"]) == list(rf["date"])
    # Kronos was shown only the context: last context timestamp is the origin,
    # and the first forecast timestamp is after it.
    for call, origin in zip(pred.seen, k["date"]):
        assert call["n"] == 30
        assert call["x_max"] == pd.Timestamp(origin)
        assert call["y_min"] > pd.Timestamp(origin)
        assert call["y_len"] == 10
        assert "close" in call["columns"]
    # Random-forest purge: every training label was known on the origin.
    assert (rf["train_last_target_date"] <= rf["date"]).all()
    assert (rf["target_date"] > rf["date"]).all()
    # Same prices scored by both models.
    assert np.allclose(k["actual_future_price"], rf["actual_future_price"])
    assert np.allclose(k["current_price"], rf["current_price"])
    # Flat fake forecast: predicted close equals current close, so return is 0.
    assert np.allclose(k["predicted_return"], 0.0)
    # Non-overlapping label windows (step == horizon).
    pos = {pd.Timestamp(d): i for i, d in enumerate(PR.prices_to_kronos(prices).index)}
    dates = [pd.Timestamp(d) for d in k["date"]]
    for a, b in zip(dates, dates[1:]):
        assert pos[b] - pos[a] >= 10


def test_summary_pairs_on_ticker_not_just_date():
    idx = pd.bdate_range("2024-01-01", periods=1)
    def row(ticker, pred):
        return {
            "date": idx[0], "ticker": ticker, "current_price": 100.0,
            "actual_future_price": 110.0, "actual_future_return": 0.1,
            "actual_direction": 1, "predicted_price": pred,
            "predicted_return": pred / 100.0 - 1.0, "prediction": int(pred > 100),
        }
    kronos = pd.DataFrame([row("AAA", 110.0), row("BBB", 200.0)])
    forest = pd.DataFrame([row("AAA", 100.0), row("BBB", 110.0)])
    s = C.summarize_pair(kronos, forest)
    assert s["n_paired"] == 2
    assert s["kronos_closer_count"] == 1
    assert s["random_forest_closer_count"] == 1


def test_summarize_counts_which_forecast_is_closer():
    idx = pd.bdate_range("2024-01-01", periods=3)
    common = {
        "date": idx,
        "ticker": "ZZ",
        "current_price": [100.0, 100.0, 100.0],
        "actual_future_price": [110.0, 90.0, 100.0],
        "actual_future_return": [0.1, -0.1, 0.0],
        "actual_direction": [1, 0, 0],
    }
    kronos = pd.DataFrame({
        **common,
        "predicted_price": [110.0, 100.0, 130.0],  # errors 0, 10, 30
        "predicted_return": [0.1, 0.0, 0.3],
        "prediction": [1, 0, 1],
    })
    forest = pd.DataFrame({
        **common,
        "predicted_price": [100.0, 90.0, 100.0],  # errors 10, 0, 0
        "predicted_return": [0.0, -0.1, 0.0],
        "prediction": [0, 0, 0],
        "reg_prediction": [0, 0, 0],
        "prob_up": [0.4, 0.4, 0.4],
    })
    s = C.summarize_pair(kronos, forest)
    assert s["n_paired"] == 3
    assert s["kronos_closer_count"] == 1
    assert s["random_forest_closer_count"] == 2
    assert s["tie_count"] == 0
    assert s["kronos"]["mae"] == 40.0 / 3.0
    assert s["random_forest"]["mae"] == 10.0 / 3.0
    assert s["random_forest"]["regressor_direction_accuracy"] == 2.0 / 3.0
