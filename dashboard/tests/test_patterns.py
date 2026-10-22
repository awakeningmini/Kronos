import numpy as np
import pandas as pd
import patterns as PT
import features as F
from conftest import make_prices


def test_engines_run_and_shapes():
    p = make_prices(400)
    for eng in ("pandas", "auto"):
        d, h, used = PT.detect(p, eng)
        assert d.shape == h.shape == (400, len(PT.PATTERNS))
        assert set(np.unique(d.values)) <= {-1, 0, 1}


def test_bullish_engulfing_pandas():
    idx = pd.bdate_range("2020-01-01", periods=15)
    o = np.linspace(100, 90, 15); c = o - 1.0
    o[-1], c[-1] = c[-2] - 0.5, o[-2] + 1.0   # big up day engulfs prior down day
    p = pd.DataFrame({"Open": o, "Close": c, "High": np.maximum(o, c) + 0.1,
                      "Low": np.minimum(o, c) - 0.1, "Volume": 1e6}, index=idx)
    d, h, _ = PT.detect(p, "pandas")
    assert d["Engulfing"].iloc[-1] == 1


def test_pattern_features_no_lookahead():
    p = make_prices(600)
    full = PT.pattern_features(p, "pandas")
    cut = PT.pattern_features(p.iloc[:450], "pandas")
    pd.testing.assert_frame_equal(full.iloc[:450], cut)
    full_t = PT.pattern_features(p)
    pd.testing.assert_frame_equal(full_t.iloc[:450], PT.pattern_features(p.iloc[:450]))


def test_build_frame_with_patterns():
    p = make_prices(800)
    df, feats = F.build_ticker_frame(p, 21, "X", use_patterns=True)
    assert all(c in feats for c in PT.PATTERN_FEATURES)
    df0, feats0 = F.build_ticker_frame(p, 21, "X")
    assert not any(c in feats0 for c in PT.PATTERN_FEATURES)


def test_forward_stats_baseline():
    fs = PT.forward_stats(make_prices(500))
    assert fs.iloc[0]["Pattern"] == "All days (baseline)"
    assert fs.iloc[0]["n 5d"] == 495
