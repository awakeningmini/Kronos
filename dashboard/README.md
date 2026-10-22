# Kronos stock research dashboard

A local Streamlit app for **personal research**. It keeps the useful pieces of the
previous Yahoo-Finance random-forest dashboard and adds [Kronos](https://github.com/shiyu-coder/Kronos)
candle forecasts on CPU.

> **Not financial advice.** Nothing here is a recommendation to buy or sell.
> `yfinance` is an unofficial way to read Yahoo Finance. It can break or rate-limit
> without warning. Kronos was pretrained on historical candles from many exchanges,
> so a backtest on those same years is **not** a clean out-of-sample test.

The upstream Kronos package (`model/`, `examples/`, and the rest of the repo) is
unchanged. This folder only calls it.

## What you can do

- Keep a ticker list in `tickers.json`.
- Random-forest horizon forecast (default research horizon is 21 trading days;
  126 is about 6 months) with the purged walk-forward gap: a training row is used
  only when its outcome date is already known on the test date.
- Optional market context: SPY, QQQ, VIX, the 10-year yield (`^TNX`), and a sector ETF.
- Optional CSV of extra columns. Values are forward-filled from their date, never backward.
- Candlestick chart, a Kronos forecast path, and a 10th–90th percentile band from
  several sampled paths.
- Candle-pattern tab (TA-Lib if installed, otherwise the built-in detector).
- A paired walk-forward that scores Kronos and the random forest on the **same**
  origin dates, with a flat-price baseline next to both.

Kronos-mini (about 4.1M parameters) is the default. The sidebar can switch to
Kronos-small or Kronos-base. The app forces **CPU** even if a GPU is present.

## Python version

Use **Python 3.14** on Windows if you have it. PyTorch 2.9 and later publish
Windows CPU wheels for CPython 3.14 (`cp314`). Pandas, scikit-learn, Streamlit,
and the other libraries in `requirements.txt` are installed from PyPI in the same
environment. This repo was executed on CPython **3.12.3** with `torch 2.14.1+cpu`
because that was the interpreter on the build machine. No dependency here requires
dropping to 3.12.

If `py -3.14` is not found, install Python 3.14 from [python.org](https://www.python.org/downloads/)
and tick **Add python.exe to PATH**, or create the venv with `py -3.12` instead.

## Setup on Windows 11 (PowerShell)

These commands call the venv Python by path. They do **not** run `Activate.ps1`,
so the execution-policy block on activation scripts does not matter.

```powershell
cd C:\path\to\Kronos\dashboard
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

If PowerShell says `.venv\Scripts\python` is not recognized, prefix `.\`:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

The browser opens at <http://localhost:8501>. The first Kronos run downloads
Kronos-mini from Hugging Face (public weights, no API key). Later runs reuse the cache.

Stop the app with **Ctrl+C** in that window.

### Tests (no internet)

```powershell
.venv\Scripts\python -m pytest
```

### Short Kronos vs random-forest backtest

Downloads TSLA and AAPL (plus SPY, QQQ, VIX, `^TNX`, and sector ETFs) and writes
`results/short_backtest.json`. The horizon is chosen from CPU timing **before**
any accuracy is computed. See the rule printed at the start of the run.

```powershell
.venv\Scripts\python run_short_backtest.py
```

## How the comparison avoids leakage

On an origin date T:

- **Random forest.** Training rows must have `date < T` and `target_date <= T`.
  The answer for a row is the close `horizon` trading days later, so this drops
  roughly the last `horizon` sessions before T. That is the same purge as the
  previous dashboard.
- **Kronos.** The context is the `lookback` bars ending on T, inclusive. The model
  is asked for the next `horizon` trading days. Those future prices are not in
  the input. Kronos is not fine-tuned on your walk-forward.
- **Step.** Origins are `horizon` trading days apart by default, so one origin's
  label window does not overlap the next origin's label window.

Kronos still only sees that ticker's OHLCV. With market context on, the forest
also sees SPY, QQQ, VIX, the 10-year yield, and a sector ETF. The scorecard
shows a flat-price forecast (predict no change) so a lower MAE than the other
model can still be worse than guessing "price stays here."

`sample_count=1` in the shipped backtest script is **one random draw**, seeded,
not a large Monte Carlo average. The chart's band uses several draws because
the upstream `predict` method averages samples internally and does not return them.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `data.py` | yfinance downloads, `tickers.json`, CSV check |
| `features.py` | Random-forest inputs and the horizon target |
| `rf_model.py` | Forests, purge, monthly backtest, signals |
| `pipeline.py` | End-to-end random-forest run |
| `patterns.py` | Candlestick patterns |
| `prepare.py` | Yahoo bars → Kronos columns, origin dates, context/future gap |
| `kronos_adapter.py` | Load upstream Kronos on CPU, sample paths for the band |
| `compare.py` | Paired walk-forward metrics |
| `run_short_backtest.py` | TSLA and AAPL proof run |
| `tests/` | Data-prep and backtest tests (offline) |
| `tickers.json` | Saved ticker list |

## Measured on the build machine (2026-10-22)

These numbers are copied from the JSON files produced by `run_short_backtest.py`.
They are not a forecast of your Windows PC and they are not financial advice.
Machine: Linux, 4 CPU cores, CPython 3.12.3, torch 2.14.1+cpu, Kronos-mini,
lookback 256, horizon 126 trading days, `sample_count=1`, seed 42. The random
forest had 100 trees, at least 750 purged rows, 3 years of warm-up, and market
context (TSLA sector ETF XLY, AAPL sector ETF XLK).

CPU time for one 126-day forecast was about **1.0 second** (TSLA origins in
`results/short_backtest.json` averaged 1.00s, range 0.99–1.01s). A 4-path
uncertainty band on the latest AAPL bars took **4.09 seconds**. Loading mini
took about **2.1 seconds** once the weights were cached. A 21-day probe took
0.17s and a 63-day probe took 0.47s, which is why the timing rule locked the
horizon at 126.

### Three origins (the timing rule)

`results/short_backtest.json`. Origins 2025-04-22, 2025-10-21, 2026-04-23.
The last target date is 2026-10-22, the same day the file was written, so that
bar may be an unfinished session.

| | Kronos MAE | Forest MAE | Flat MAE | Kronos up/down | Forest classifier | Always-up rate |
|---|---:|---:|---:|---:|---:|---:|
| TSLA (n=3) | $105.52 | $98.27 | $91.75 | 100% | 33% | 67% |
| AAPL (n=3) | $64.46 | $34.66 | $49.47 | 33% | 100% | 100% |
| Both (n=6) | $84.99 | $66.47 | $70.61 | 67% | 67% | 83% |

"Forest classifier" is P(up) ≥ 0.5. The price MAE uses the forest's return
model, whose sign was right on 67% of TSLA origins, 100% of AAPL origins, and
83% pooled (the same as the always-up rate). On these six forecasts the forest
was closer on 5 origins and Kronos on 1. Pooled average absolute percent error
was 23.7% (Kronos), 17.8% (forest), 19.1% (flat). Neither up/down call beat
the always-up rate on the pooled set.

### Eight origins (follow-up, same horizon)

`results/longer_backtest.json`. Same settings, origin count raised to 8 after
the short run showed each forecast cost about a second. Origins run from
2022-10-14 through 2026-04-23 (16 forecasts). This was not the pre-registered
count.

| | Kronos MAE | Forest MAE | Flat MAE | Kronos up/down | Forest classifier | Always-up rate |
|---|---:|---:|---:|---:|---:|---:|
| TSLA (n=8) | $61.04 | $77.81 | $69.95 | 75% | 50% | 62.5% |
| AAPL (n=8) | $35.29 | $32.86 | $36.76 | 75% | 75% | 75% |
| Both (n=16) | $48.16 | $55.34 | $53.35 | 75% | 62.5% | 68.8% |

Closer-origin count was 8–8. Kronos had the lower pooled MAE. The forest was
slightly closer on AAPL. The forest return-sign hit rate was 62.5% on TSLA
(equal to always-up), 75% on AAPL (equal to always-up), and 68.8% pooled
(equal to always-up). Kronos up/down beat the always-up rate on TSLA and on
the pooled set. On AAPL both models matched that rate.

Read both tables together. Three recent windows favor the forest. Eight windows
favor Kronos on TSLA and leave AAPL roughly tied. Sixteen non-overlapping
six-month draws, one Kronos sample each, is still a small test, and Kronos may
have seen 2022–2025 bars in pretraining. It does not justify retiring the
random forest.

## Troubleshooting

- **`py` is not recognized.** Re-run the Python installer and tick Add to PATH.
- **Torch install is huge or tries to use CUDA.** Install the CPU wheel first, with the `--index-url` command above, then install `requirements.txt`. Pip will keep that wheel if it is new enough.
- **`No data downloaded`.** Check the symbol on Yahoo Finance, wait a few minutes, then `.venv\Scripts\python -m pip install -U yfinance`.
- **Not enough history.** Move the start date earlier or lower the minimum training years. The forest wants 750 purged training rows.
- **Kronos is slow.** Stay on mini, shorten the horizon, and lower the number of chart paths or backtest origins. 126 steps means 126 autoregressive steps per path.
- **TA-Lib failed to install.** Leave it uninstalled. The pattern tab uses the built-in detector. A commented line in `requirements.txt` is the optional install.
