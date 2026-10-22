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

## Troubleshooting

- **`py` is not recognized.** Re-run the Python installer and tick Add to PATH.
- **Torch install is huge or tries to use CUDA.** Install the CPU wheel first, with the `--index-url` command above, then install `requirements.txt`. Pip will keep that wheel if it is new enough.
- **`No data downloaded`.** Check the symbol on Yahoo Finance, wait a few minutes, then `.venv\Scripts\python -m pip install -U yfinance`.
- **Not enough history.** Move the start date earlier or lower the minimum training years. The forest wants 750 purged training rows.
- **Kronos is slow.** Stay on mini, shorten the horizon, and lower the number of chart paths or backtest origins. 126 steps means 126 autoregressive steps per path.
- **TA-Lib failed to install.** Leave it uninstalled. The pattern tab uses the built-in detector. A commented line in `requirements.txt` is the optional install.
