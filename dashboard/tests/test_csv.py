import numpy as np
import pandas as pd
import pytest

import data as D
import features as F


def test_csv_validation_errors():
    with pytest.raises(ValueError, match="Date"):
        D.parse_extra_csv(b"day,x\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="numeric"):
        D.parse_extra_csv(b"Date,name\n2020-01-01,abc\n")
    with pytest.raises(ValueError, match="dates"):
        D.parse_extra_csv(b"Date,x\nhello,1\nworld,2\n")
    with pytest.raises(ValueError, match="empty"):
        D.parse_extra_csv(b"Date,x\n")


def test_csv_parse_ok():
    out = D.parse_extra_csv(b"date,Fed Rate,label\n2020-01-03,1.5,a\n2020-01-01,1.25,b\n")
    assert list(out.columns) == ["csv_Fed_Rate"]
    assert out.index.is_monotonic_increasing
    assert out.attrs["skipped_columns"] == ["label"]


def test_csv_join_is_forward_fill_only(prices):
    """A CSV value dated D may only appear on dates >= D; never earlier (no backfill)."""
    d1, d2 = prices.index[300], prices.index[310]
    # Monthly-style data on non-trading timestamps (a Saturday) also must not leak backwards
    sat = d2 + pd.Timedelta(days=(5 - d2.weekday()) % 7 or 7)
    raw = f"Date,macro\n{d1.date()},1.0\n{d2.date()},2.0\n{sat.date()},3.0\n".encode()
    csv = D.parse_extra_csv(raw)
    frame, feats = F.build_ticker_frame(prices, 126, "X", extra=[csv])
    col = frame["csv_macro"]
    assert col.loc[: prices.index[299]].isna().all()            # before first CSV date: NaN, not backfilled
    assert (col.loc[d1: prices.index[309]] == 1.0).all()
    assert col.loc[d2] == 2.0
    next_trading = prices.index[prices.index > sat][0]
    assert col.loc[prices.index[(prices.index >= d2) & (prices.index < sat)]].eq(2.0).all()
    assert col.loc[next_trading] == 3.0
    # Generic invariant: every non-NaN value equals the latest CSV value dated <= that day
    for d, v in col.dropna().items():
        assert v == csv["csv_macro"].loc[:d].iloc[-1]


def test_align_to_dates_never_backfills():
    other = pd.Series([5.0], index=[pd.Timestamp("2020-06-01")])
    dates = pd.bdate_range("2020-05-25", "2020-06-05")
    out = F.align_to_dates(other, dates)
    assert out.loc[:"2020-05-29"].isna().all()
    assert (out.loc["2020-06-01":] == 5.0).all()
