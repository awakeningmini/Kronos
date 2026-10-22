import numpy as np
import pandas as pd

import features as F
import pipeline as P
import rf_model as M
from conftest import make_prices

H = 126


def long_table(tickers=("AAA",), n=1500):
    frames, feats = {}, {}
    for i, t in enumerate(tickers):
        fr, fl = F.build_ticker_frame(make_prices(n=n, seed=i + 1, start=f"201{i}-03-01"), H, t)
        frames[t], feats[t] = fr, fl
    return F.pool_frames(frames, feats)


def test_purge_matches_position_rule():
    lab, feats = long_table()
    for test_pos in [900, 1000, 1200]:
        test_row = lab[lab["pos"] == test_pos].iloc[0]
        train = M.purged_train_set(lab, test_row["date"])
        assert (train["target_date"] <= test_row["date"]).all()
        assert train["pos"].max() == test_pos - H   # exactly position <= test_pos - HORIZON
        assert (train["date"] < test_row["date"]).all()


def test_walk_forward_has_no_training_label_after_test_date():
    lab, feats = long_table()
    res = M.walk_forward(lab, feats, min_train_years=2, n_estimators=10)
    assert len(res) > 5
    assert (res["train_last_target_date"] <= res["date"]).all()


def test_pooled_walk_forward_purges_across_tickers():
    lab, feats = long_table(("AAA", "BBB"))
    res = M.walk_forward(lab, feats, min_train_years=2, n_estimators=10)
    assert set(res["ticker"]) == {"AAA", "BBB"}
    # cutoff is the earliest test date that month, so no ticker's training label passes its test date
    assert (res["train_last_target_date"] <= res["date"]).all()


def test_pooled_dataset_construction():
    fa, la = F.build_ticker_frame(make_prices(seed=1), H, "AAA")
    fb, lb = F.build_ticker_frame(make_prices(seed=2, n=1200, start="2013-06-03"), H, "BBB",
                                  extra=[pd.DataFrame({"csv_x": [1.0]}, index=[pd.Timestamp("2010-01-01")])])
    pooled, common = F.pool_frames({"AAA": fa, "BBB": fb}, {"AAA": la, "BBB": lb})
    assert common == la                       # BBB-only column dropped
    assert "csv_x" not in common and "ticker" not in common
    assert set(pooled["ticker"]) == {"AAA", "BBB"}
    assert len(pooled) == len(F.labeled_rows(fa, la)) + len(F.labeled_rows(fb, la))
    assert pooled["date"].is_monotonic_increasing
    assert not pooled[common + ["target", "future_return"]].isna().any().any()
    assert (pooled["target_date"] > pooled["date"]).all()


def test_scorecard_and_signal():
    assert M.signal_for(0.7) == "STRONG BULLISH"
    assert M.signal_for(0.6) == "BULLISH"
    assert M.signal_for(0.5) == "NEUTRAL"
    assert M.signal_for(0.4) == "BEARISH"
    assert M.signal_for(0.3) == "STRONG BEARISH"
    res = pd.DataFrame({
        "date": pd.bdate_range("2020-01-31", periods=12, freq="BME"),
        "pos": np.arange(12) * 21,
        "actual_direction": [1] * 9 + [0] * 3,
        "prediction": [1] * 12,
        "prob_up": np.linspace(0.4, 0.9, 12),
        "actual_future_price": np.full(12, 110.0),
        "predicted_price": np.full(12, 100.0),
        "actual_future_return": np.full(12, 0.1),
    })
    s = M.scorecard(res, H)
    assert s["accuracy"] == 0.75 and s["base_rate_always_up"] == 0.75
    assert s["beats_base_rate"] is False      # ties don't count as beating
    assert s["mae"] == 10.0
    assert s["n_non_overlap_trades"] == 2     # pos 0 and 147 (>=126 trading days apart)


def test_pipeline_end_to_end_offline():
    fake = {"AAA": make_prices(seed=1, n=1400), "BBB": make_prices(seed=2, n=1400),
            "SPY": make_prices(seed=3, n=1400), "QQQ": make_prices(seed=4, n=1400),
            "^VIX": make_prices(seed=5, n=1400), "^TNX": make_prices(seed=6, n=1400),
            "XLK": make_prices(seed=7, n=1400)}
    cfg = P.Config(["AAA", "BBB", "NOPE"], start="2012-01-01", min_train_years=2, pooled=True,
                   use_market=True, backtest_trees=10, final_trees=10)

    def gp(t, s):
        if t not in fake:
            raise ValueError("No data downloaded for " + t)
        return fake[t]

    out = P.run(cfg, get_prices=gp, get_close=lambda t, s: gp(t, s)["Close"],
                sector_lookup=lambda t: ("XLK", "Technology"))
    assert "NOPE" in out["errors"]
    assert set(out["tickers"]) == {"AAA", "BBB"}
    a = out["tickers"]["AAA"]
    assert 0 <= a["forecast"]["prob_up"] <= 1
    assert "vix_level" in a["features"] and "rs_sector_60" in a["features"]
    assert a["scorecard"]["n_predictions"] > 0
    tbl = P.summary_table(out)
    assert list(tbl["Ticker"]) == ["AAA", "BBB"]


def test_pipeline_with_csvs_single_mode():
    fake = {"AAA": make_prices(seed=1, n=1400), "BBB": make_prices(seed=2, n=1400)}
    idx = fake["AAA"].index[::21]
    g = pd.DataFrame({"csv_macro": np.arange(len(idx), dtype=float)}, index=idx)
    t = pd.DataFrame({"csv_sent": np.ones(len(idx))}, index=idx)
    cfg = P.Config(["AAA", "BBB"], min_train_years=2, global_csv=g, ticker_csvs={"AAA": t},
                   backtest_trees=10, final_trees=10)
    out = P.run(cfg, get_prices=lambda k, s: fake[k])
    assert "csv_macro" in out["tickers"]["AAA"]["features"]
    assert "AAA_csv_sent" in out["tickers"]["AAA"]["features"]
    assert "AAA_csv_sent" not in out["tickers"]["BBB"]["features"]
    # pooled: per-ticker CSV columns dropped with a warning, global kept
    cfg.pooled = True
    out = P.run(cfg, get_prices=lambda k, s: fake[k])
    assert "AAA_csv_sent" not in out["tickers"]["AAA"]["features"]
    assert "csv_macro" in out["tickers"]["AAA"]["features"]
    assert any("pooled mode" in w for w in out["warnings"])
