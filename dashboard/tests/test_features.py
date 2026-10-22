import numpy as np
import pandas as pd
import pytest

import data as D
import features as F
from conftest import make_prices


def test_price_features_match_formulas(prices):
    df = F.add_price_features(prices)
    c = prices["Close"]
    i = 300
    assert df["ret_1"].iloc[i] == pytest.approx(c.iloc[i] / c.iloc[i - 1] - 1)
    assert df["ret_20"].iloc[i] == pytest.approx(c.iloc[i] / c.iloc[i - 20] - 1)
    assert df["ma_ratio_60"].iloc[i] == pytest.approx(c.iloc[i] / c.iloc[i - 59:i + 1].mean())
    rets = c.pct_change()
    assert df["vol_10"].iloc[i] == pytest.approx(rets.iloc[i - 9:i + 1].std())
    h, l = prices["High"].iloc[i], prices["Low"].iloc[i]
    assert df["hl_range"].iloc[i] == pytest.approx((h - l) / c.iloc[i])
    assert df["close_position"].iloc[i] == pytest.approx((c.iloc[i] - l) / (h - l))
    assert set(F.BASE_FEATURES) <= set(df.columns)


def test_features_do_not_use_future_prices(prices):
    """Changing prices AFTER day k must not change any feature on or before day k."""
    k = 800
    a = F.add_price_features(prices)
    p2 = prices.copy()
    p2.iloc[k + 1:] *= 3.0
    b = F.add_price_features(p2)
    pd.testing.assert_frame_equal(a[F.BASE_FEATURES].iloc[:k + 1], b[F.BASE_FEATURES].iloc[:k + 1])


def test_target_and_target_date(prices):
    h = 126
    df = F.add_target(prices, h)
    i = 100
    assert df["future_close"].iloc[i] == prices["Close"].iloc[i + h]
    assert df["target_date"].iloc[i] == prices.index[i + h]
    assert df["target"].iloc[i] == float(prices["Close"].iloc[i + h] > prices["Close"].iloc[i])
    assert df["future_close"].iloc[-h:].isna().all()
    assert df["target"].iloc[-h:].isna().all()


def test_zero_volume_index_does_not_drop_rows(prices):
    p = prices.copy()
    p["Volume"] = 0
    df = F.add_price_features(p)
    assert (df["volume_change_5"] == 0).all()


def test_market_features_aligned_without_lookahead(prices):
    spy = make_prices(seed=5)["Close"]
    vix = pd.Series(np.linspace(10, 40, len(spy)), index=spy.index)
    mf = F.market_feature_frame({"spy": spy, "vix": vix})
    frame, feats = F.build_ticker_frame(prices, 126, "X", mf)
    assert "spy_ret_20" in feats and "rs_spy_60" in feats and "vix_level" in feats
    d = prices.index[400]
    assert frame.loc[d, "vix_level"] == vix.loc[d]
    assert frame.loc[d, "rs_spy_20"] == pytest.approx(frame.loc[d, "ret_20"] - mf.loc[d, "spy_ret_20"])


def test_sector_falls_back_to_spy_before_etf_existed(prices):
    spy = make_prices(seed=5)["Close"]
    sector = make_prices(seed=6)["Close"].iloc[700:]  # ETF "launched" later
    mf = F.market_feature_frame({"spy": spy})
    sf = F.sector_feature_frame(sector)
    frame, feats = F.build_ticker_frame(prices, 126, "X", mf, sf)
    d_early = prices.index[300]
    assert frame.loc[d_early, "sector_ret_20"] == frame.loc[d_early, "spy_ret_20"]
    d_late = prices.index[1000]
    assert frame.loc[d_late, "sector_ret_20"] == pytest.approx(sf.loc[d_late, "sector_ret_20"])


def test_clean_ticker():
    assert D.clean_ticker(" tsla ") == "TSLA"
    assert D.clean_ticker("brk-b") == "BRK-B"
    assert D.clean_ticker("^gspc") == "^GSPC"
    with pytest.raises(ValueError):
        D.clean_ticker("not a ticker!")


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    assert D.load_settings(p)["tickers"] == ["TSLA"]
    D.save_settings({"tickers": ["TSLA", "AAPL"], "sector_overrides": {"AAPL": "XLK"}}, p)
    s = D.load_settings(p)
    assert s["tickers"] == ["TSLA", "AAPL"] and s["sector_overrides"]["AAPL"] == "XLK"
    p.write_text("{garbage")
    assert D.load_settings(p)["tickers"] == ["TSLA"]
