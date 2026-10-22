"""Local stock research dashboard: random forest plus Kronos candle forecasts.

Personal research only. Not financial advice.

From this folder, without activating the virtual environment:

    .venv\\Scripts\\python -m streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as D
import features as F
import patterns as PT
import pipeline as P
import prepare as PR
import rf_model as M

st.set_page_config(page_title="Kronos Stock Research", page_icon="📈", layout="wide")

DISCLAIMER = (
    "**Personal research only. Not financial advice.** Prices come from Yahoo Finance via "
    "`yfinance`, an unofficial tool that can break or get rate-limited. Kronos is a pretrained "
    "candle model run on CPU. A backtest can overlap the years it was trained on, so a higher "
    "score here is not proof it will beat a random forest in the future."
)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_prices(ticker: str, start: str) -> pd.DataFrame:
    return D.download_prices(ticker, start)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_close(symbol: str, start: str) -> pd.Series:
    return D.download_close(symbol, start)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_sector(ticker: str):
    return D.lookup_sector_etf(ticker)


@st.cache_resource(show_spinner=False)
def cached_predictor(model_key: str):
    from kronos_adapter import load_predictor
    return load_predictor(model_key, device="cpu")


@st.cache_data(show_spinner=False, max_entries=20)
def cached_rf(tickers: tuple, start: str, horizon: int, min_years: float, pooled: bool,
              market: bool, overrides: tuple, global_csv: bytes | None,
              ticker_csvs: tuple, trees: int, data_day: str, use_patterns: bool = False):
    cfg = P.Config(
        tickers=list(tickers), start=start, horizon=horizon, min_train_years=min_years,
        pooled=pooled, use_market=market, sector_overrides=dict(overrides),
        global_csv=D.parse_extra_csv(global_csv) if global_csv else None,
        ticker_csvs={t: D.parse_extra_csv(b) for t, b in ticker_csvs},
        backtest_trees=trees, use_patterns=use_patterns,
    )
    out = P.run(cfg, get_prices=cached_prices, get_close=cached_close, sector_lookup=cached_sector)
    out.pop("config", None)
    return out


def _pct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.1%}"


def _money(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"${v:,.2f}"


def _market_frame(start: str, use_market: bool):
    warnings = []
    if not use_market:
        return None, warnings
    market = {}
    for key, sym in D.MARKET_SYMBOLS.items():
        try:
            market[key] = cached_close(sym, start)
        except Exception as exc:
            warnings.append(f"Market series {sym} unavailable ({exc}); skipped.")
    return F.market_feature_frame(market), warnings


def _extras_for(ticker: str, global_csv: bytes | None, ticker_csvs: tuple):
    extras = []
    if global_csv:
        extras.append(D.parse_extra_csv(global_csv))
    for name, raw in ticker_csvs:
        if name == ticker:
            extras.append(D.parse_extra_csv(raw).add_prefix(f"{ticker}_".replace("^", "")))
    return extras


if "settings" not in st.session_state:
    st.session_state.settings = D.load_settings()
if "csvs" not in st.session_state:
    st.session_state.csvs = {}
settings = st.session_state.settings


def persist():
    D.save_settings(settings)


with st.sidebar:
    st.header("1. Tickers")
    with st.form("add_ticker", clear_on_submit=True):
        new = st.text_input("Add a ticker", placeholder="e.g. AAPL, NVDA, BRK-B")
        if st.form_submit_button("Add"):
            try:
                t = D.clean_ticker(new)
                if t in settings["tickers"]:
                    st.info(f"{t} is already in the list.")
                else:
                    settings["tickers"].append(t)
                    persist()
                    st.success(f"Added {t}.")
            except ValueError as e:
                st.error(str(e))
    for t in list(settings["tickers"]):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{t}**")
        if c2.button("x", key=f"rm_{t}", help=f"Remove {t}", disabled=len(settings["tickers"]) <= 1):
            settings["tickers"].remove(t)
            settings["sector_overrides"].pop(t, None)
            st.session_state.csvs.pop(t, None)
            persist()
            st.rerun()
    st.caption("Saved in `tickers.json` next to this app.")

    st.header("2. Shared settings")
    start = st.date_input(
        "History start date", value=dt.date(2012, 1, 1),
        min_value=dt.date(1990, 1, 1), max_value=dt.date.today(),
    )
    horizon = st.number_input(
        "Horizon (trading days)", 5, 252, 21, step=1,
        help="Random-forest target and Kronos forecast length. 21 ≈ 1 month, 126 ≈ 6 months. "
             "Kronos generates one candle per day, so 126 is much slower on CPU.",
    )
    min_years = st.number_input("Min training years before first backtest", 1.0, 15.0, 3.0, step=0.5)

    st.header("3. Random forest")
    mode = st.radio(
        "Training mode",
        ["Single ticker (one model per stock)", "Pooled (one model trained on all tickers)"],
    )
    pooled = mode.startswith("Pooled")
    market = st.toggle("Add market context (SPY, QQQ, VIX, 10-year yield, sector ETF)", value=True)
    if market:
        with st.expander("Sector ETF per ticker"):
            opts = ["Auto", "None"] + D.SECTOR_ETFS
            for t in settings["tickers"]:
                cur = settings["sector_overrides"].get(t, "Auto")
                choice = st.selectbox(t, opts, index=opts.index(cur) if cur in opts else 0, key=f"sec_{t}")
                if choice != cur:
                    settings["sector_overrides"][t] = choice
                    persist()
    use_patterns = st.toggle("Add candlestick-pattern features to the random forest", value=False)
    trees = st.select_slider("Backtest trees", [50, 100, 200], value=100)

    st.header("4. Kronos (CPU)")
    from kronos_adapter import MODELS
    model_key = st.selectbox(
        "Model",
        ["mini", "small", "base"],
        format_func=lambda k: MODELS[k]["label"],
        help="mini is the default. small and base are slower and use a 512-bar context.",
    )
    max_ctx = MODELS[model_key]["max_context"]
    lookback = st.number_input(
        "Lookback (trading days)", 32, int(max_ctx), min(256, max_ctx), step=16,
        key=f"lookback_{model_key}",
    )
    n_paths = st.number_input(
        "Chart sample paths (uncertainty band)", 1, 8, 4, step=1,
        help="Each path is one sample. The band is the 10th to 90th percentile of close. "
             "Upstream Kronos averages samples inside one call, so the band is several calls.",
    )
    temperature = st.number_input("Temperature", 0.1, 2.0, 1.0, step=0.1)
    top_p = st.number_input("top_p", 0.1, 1.0, 0.9, step=0.05)
    bt_samples = st.number_input(
        "Backtest samples averaged per origin", 1, 4, 1, step=1,
        help="Passed to Kronos as sample_count. 1 is one stochastic draw and is the fastest.",
    )
    max_origins = st.number_input("Walk-forward origins per ticker", 1, 36, 6, step=1)
    step_in = st.number_input(
        "Origin step (0 = same as horizon)", 0, 252, 0, step=1,
        help="Trading days between origins. The default leaves a full horizon between origins "
             "so label windows do not overlap.",
    )
    seed = st.number_input("Random seed", 0, 10_000, 42, step=1)

    st.header("5. Extra data (CSV)")
    up = st.file_uploader("CSV with a Date column and number columns", type=["csv"])
    scope = st.selectbox("Use it for", ["All tickers"] + settings["tickers"])
    if up is not None and st.button("Add this CSV"):
        raw = up.getvalue()
        try:
            parsed = D.parse_extra_csv(raw)
            key = "__ALL__" if scope == "All tickers" else scope
            st.session_state.csvs[key] = (up.name, raw)
            st.success(f"Added {up.name}: {len(parsed)} rows, columns {list(parsed.columns)}")
        except ValueError as e:
            st.error(f"CSV problem: {e}")
    for key, (name, _) in list(st.session_state.csvs.items()):
        c1, c2 = st.columns([3, 1])
        c1.caption(f"{name} -> {'all tickers' if key == '__ALL__' else key}")
        if c2.button("x", key=f"rmcsv_{key}"):
            st.session_state.csvs.pop(key)
            st.rerun()

    run_rf = st.button("Run random forest", type="primary", width="stretch")
    run_kronos = st.button("Kronos forecast (latest bars)", width="stretch")
    run_cmp = st.button("Compare on walk-forward backtest", width="stretch")
    st.caption("Kronos runs on CPU. A long horizon or many origins can take several minutes.")

# ----------------------------------------------------------------------------
st.title("Stock research: random forest and Kronos")
st.warning(DISCLAIMER)

csvs = st.session_state.csvs
start_s = str(start)
H = int(horizon)
rf_key = dict(
    tickers=tuple(settings["tickers"]), start=start_s, horizon=H,
    min_years=float(min_years), pooled=pooled, market=market,
    overrides=tuple(sorted(settings["sector_overrides"].items())),
    global_csv=csvs["__ALL__"][1] if "__ALL__" in csvs else None,
    ticker_csvs=tuple(sorted((k, v[1]) for k, v in csvs.items() if k != "__ALL__" and k in settings["tickers"])),
    trees=int(trees), data_day=str(dt.date.today()), use_patterns=bool(use_patterns),
)
key_hash = hashlib.md5(repr(rf_key).encode()).hexdigest()

if run_rf:
    st.session_state.last_run = rf_key
    st.session_state.last_hash = key_hash

if run_kronos:
    from kronos_adapter import forecast_latest
    forecasts = {}
    errors = {}
    with st.spinner("Loading Kronos on CPU and forecasting each ticker…"):
        try:
            predictor = cached_predictor(model_key)
        except Exception as exc:
            st.error(f"Could not load Kronos: {exc}")
            predictor = None
        if predictor is not None:
            for t in settings["tickers"]:
                try:
                    prices = cached_prices(t, start_s)
                    kprices = PR.prices_to_kronos(prices)
                    forecasts[t] = forecast_latest(
                        predictor, kprices, lookback=int(lookback), pred_len=H,
                        n_paths=int(n_paths), T=float(temperature), top_p=float(top_p),
                        seed=int(seed),
                    )
                    forecasts[t]["prices"] = prices
                except Exception as exc:
                    errors[t] = str(exc)
    st.session_state.kronos_forecasts = forecasts
    st.session_state.kronos_errors = errors
    st.session_state.kronos_settings = {
        "model": model_key, "lookback": int(lookback), "horizon": H,
        "n_paths": int(n_paths), "T": float(temperature), "top_p": float(top_p),
    }

if run_cmp:
    import compare as C
    step = H if int(step_in) == 0 else int(step_in)
    reports = {}
    errors = {}
    warnings = []
    with st.spinner("Walk-forward comparison on CPU. Each origin is a full Kronos forecast…"):
        try:
            predictor = cached_predictor(model_key)
            market_feats, warnings = _market_frame(start_s, market)
            global_csv = csvs["__ALL__"][1] if "__ALL__" in csvs else None
            ticker_csvs = tuple(sorted(
                (k, v[1]) for k, v in csvs.items() if k != "__ALL__" and k in settings["tickers"]
            ))
            for t in settings["tickers"]:
                try:
                    prices = cached_prices(t, start_s)
                    sector_feats = None
                    sector_note = None
                    if market:
                        etf, why = P.resolve_sector(t, settings["sector_overrides"], cached_sector)
                        sector_note = (etf, why)
                        if etf:
                            try:
                                sector_feats = F.sector_feature_frame(cached_close(etf, start_s))
                            except Exception as exc:
                                warnings.append(f"{t}: sector ETF {etf} unavailable ({exc}).")
                    result = C.run_ticker(
                        t, prices, predictor,
                        horizon=H, lookback=int(lookback), step=step,
                        max_origins=int(max_origins),
                        market_feats=market_feats, sector_feats=sector_feats,
                        extra=_extras_for(t, global_csv, ticker_csvs),
                        use_patterns=bool(use_patterns),
                        n_estimators=int(trees),
                        min_train_rows=M.MIN_TRAIN_ROWS,
                        min_train_years=float(min_years),
                        T=float(temperature), top_p=float(top_p),
                        sample_count=int(bt_samples), seed=int(seed),
                    )
                    result["sector"] = sector_note
                    result["prices"] = prices
                    reports[t] = result
                except Exception as exc:
                    errors[t] = str(exc)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    st.session_state.comparison = reports
    st.session_state.comparison_errors = errors
    st.session_state.comparison_warnings = warnings
    st.session_state.comparison_meta = {
        "horizon": H, "lookback": int(lookback), "step": step,
        "model": model_key, "sample_count": int(bt_samples), "seed": int(seed),
    }

rf_result = None
if "last_run" in st.session_state:
    if st.session_state.get("last_hash") != key_hash:
        st.caption("Random-forest settings changed since the last run. Press **Run random forest** to refresh.")
    with st.spinner("Random-forest download and purged monthly backtest…"):
        try:
            rf_result = cached_rf(**st.session_state.last_run)
        except Exception as exc:
            st.error(f"Random forest failed: {exc}")

if rf_result is None and "kronos_forecasts" not in st.session_state and "comparison" not in st.session_state:
    st.info(
        "Pick tickers in the sidebar, then run the random forest, a Kronos forecast, "
        "or the walk-forward comparison. The first Kronos run downloads the mini weights."
    )

tab_cmp, tab_one, tab_k, tab_vs, tab_pat, tab_help = st.tabs(
    ["All tickers", "Ticker detail", "Kronos forecast", "Kronos vs forest", "Candle patterns", "How it works"]
)

with tab_cmp:
    st.subheader(f"Random forest — horizon {H} trading days")
    if not rf_result or not rf_result.get("tickers"):
        st.info("No random-forest run yet.")
    else:
        for t, err in rf_result["errors"].items():
            st.error(f"**{t}** skipped: {err}")
        for w in rf_result["warnings"]:
            st.warning(w)
        tbl = P.summary_table(rf_result)
        fmt = {c: "{:.1%}" for c in [
            "P(higher)", "Exp. return", "Confidence", "Backtest accuracy", "Always-up rate",
            "Avg % error", "Model signal return", "Buy & hold return",
        ]}
        fmt.update({"Price": "${:,.2f}", "Est. price": "${:,.2f}", "ROC-AUC": "{:.3f}"})
        st.dataframe(tbl.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
        st.caption(
            "“Beats base rate?” is monthly purged-walk-forward accuracy versus always guessing up. "
            + ("Pooled model." if st.session_state.last_run["pooled"] else "One model per ticker.")
        )

with tab_one:
    if not rf_result or not rf_result.get("tickers"):
        st.info("Run the random forest to see this tab.")
    else:
        tick = st.selectbox("Ticker", list(rf_result["tickers"]), key="rf_tick")
        r = rf_result["tickers"][tick]
        fcast, sc, res = r["forecast"], r["scorecard"], r["backtest"]
        if fcast:
            c = st.columns(6)
            c[0].metric(f"P(higher in {H}d)", _pct(fcast["prob_up"]))
            c[1].metric("Estimated price", _money(fcast["predicted_price"]), f"{fcast['expected_return']:+.1%}")
            c[2].metric("Current price", _money(fcast["current_price"]))
            c[3].metric("Signal", fcast["signal"])
            c[4].metric("Confidence", _pct(fcast["confidence"]))
            c[5].metric("Data as of", str(pd.Timestamp(fcast["as_of"]).date()))
        st.caption(
            f"Trained on: {r['trained_on']} · {r['n_features']} features · "
            f"labeled {r['labeled_range'][0].date()} → {r['labeled_range'][1].date()}"
        )
        if not sc:
            st.info("No monthly backtest for this ticker yet. Try an earlier start date.")
        else:
            if sc["beats_base_rate"]:
                st.success(
                    f"Accuracy {_pct(sc['accuracy'])} beat the always-up rate {_pct(sc['base_rate_always_up'])}."
                )
            else:
                st.error(
                    f"Accuracy {_pct(sc['accuracy'])} did not beat always guessing up "
                    f"({_pct(sc['base_rate_always_up'])})."
                )
            left, right = st.columns(2)
            left.markdown("**Direction**")
            left.table(pd.DataFrame({
                "Model": [
                    _pct(sc["accuracy"]), _pct(sc["precision"]),
                    "—" if pd.isna(sc["roc_auc"]) else f"{sc['roc_auc']:.3f}",
                    _pct(sc["pct_predicted_up"]),
                ],
                "Baseline": [
                    _pct(sc["base_rate_always_up"]) + " (always up)",
                    _pct(sc["base_rate_always_up"]),
                    "0.500",
                    "100% (always up)",
                ],
            }, index=["Accuracy", "Precision", "ROC-AUC", "Share called up"]))
            right.markdown("**Price**")
            right.table(pd.DataFrame({"Value": {
                "MAE": _money(sc["mae"]),
                "RMSE": _money(sc["rmse"]),
                "Avg % error": _pct(sc["avg_pct_error"]),
                "Signal return": _pct(sc.get("model_signal_return")),
                "Buy & hold": _pct(sc.get("buy_hold_return")),
            }}))
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=res.index, y=res["prob_up"], name="P(higher)"))
            fig1.add_hline(y=0.5, line_dash="dash")
            fig1.update_layout(title=f"{tick} monthly P(higher)", yaxis_range=[0, 1], height=380)
            st.plotly_chart(fig1, width="stretch")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.index, y=res["actual_future_price"], name="Actual future"))
            fig2.add_trace(go.Scatter(x=res.index, y=res["predicted_price"], name="Predicted"))
            fig2.update_layout(title=f"{tick} monthly price backtest", height=380)
            st.plotly_chart(fig2, width="stretch")
        imp = r["importance"].head(15).sort_values()
        fig4 = go.Figure(go.Bar(x=imp.values, y=imp.index.astype(str), orientation="h"))
        fig4.update_layout(title="Feature importance", height=max(360, 26 * len(imp)))
        st.plotly_chart(fig4, width="stretch")
        if len(res):
            with st.expander("Monthly backtest rows"):
                st.dataframe(res, width="stretch")
                st.download_button("Download CSV", res.to_csv().encode(), f"{tick}_rf_backtest.csv", "text/csv")

with tab_k:
    forecasts = st.session_state.get("kronos_forecasts") or {}
    for t, err in (st.session_state.get("kronos_errors") or {}).items():
        st.error(f"**{t}**: {err}")
    if not forecasts:
        st.info("Press **Kronos forecast** in the sidebar. The first run downloads Kronos-mini.")
    else:
        meta = st.session_state.get("kronos_settings") or {}
        tick = st.selectbox("Ticker", list(forecasts), key="k_tick")
        fc = forecasts[tick]
        prices = fc["prices"]
        c = st.columns(5)
        c[0].metric("Mean forecast price", _money(fc["terminal_mean"]), f"{fc['expected_return']:+.1%}")
        c[1].metric("10th–90th percentile", f"{_money(fc['terminal_p10'])} – {_money(fc['terminal_p90'])}")
        c[2].metric("Paths finishing up", _pct(fc["up_rate"]) if fc["n_paths"] > 1 else "one path")
        c[3].metric("Last close", _money(fc["current_price"]))
        c[4].metric("CPU seconds", f"{fc['seconds']:.1f}")
        st.caption(
            f"Model {meta.get('model')} · lookback {meta.get('lookback')} · "
            f"{fc['n_paths']} path(s) · temperature {meta.get('T')} · top_p {meta.get('top_p')} · "
            f"as of {pd.Timestamp(fc['as_of']).date()} · device CPU. "
            "Future dates are weekdays, not exchange holidays. "
            "The shaded band is the 10th–90th percentile of sampled closes. "
            "With one path the band is just that path."
        )
        window = st.select_slider("History shown (trading days)", [60, 120, 250], value=120, key="k_win")
        hist = prices.iloc[-int(window):]
        mean = fc["mean"]
        band = fc["band"]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=tick,
        ))
        fig.add_trace(go.Candlestick(
            x=mean.index, open=mean["open"], high=mean["high"], low=mean["low"], close=mean["close"],
            name="Kronos mean", increasing_line_color="#f4a261", decreasing_line_color="#e76f51",
        ))
        fig.add_trace(go.Scatter(
            x=band.index, y=band["close_p90"], line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=band.index, y=band["close_p10"], fill="tonexty", name="Close 10–90%",
            line=dict(width=0), fillcolor="rgba(244,162,97,0.25)",
        ))
        fig.update_layout(
            title=f"{tick} Kronos candle forecast", height=560, xaxis_rangeslider_visible=False,
            yaxis_title="Price",
        )
        st.plotly_chart(fig, width="stretch")
        show = mean.copy()
        show = show.join(band[["close_p10", "close_p90"]])
        st.dataframe(show, width="stretch")
        st.download_button(
            "Download forecast CSV", show.to_csv().encode(), f"{tick}_kronos_forecast.csv", "text/csv",
            key="dl_k",
        )

with tab_vs:
    reports = st.session_state.get("comparison") or {}
    for t, err in (st.session_state.get("comparison_errors") or {}).items():
        st.error(f"**{t}**: {err}")
    for w in st.session_state.get("comparison_warnings") or []:
        st.warning(w)
    if not reports:
        st.info(
            "Press **Compare on walk-forward backtest**. Both models are scored on the same origin "
            "dates. The forest is retrained with a purge gap. Kronos only receives bars up to the origin."
        )
    else:
        meta = st.session_state.get("comparison_meta") or {}
        st.caption(
            f"Model {meta.get('model')} · horizon {meta.get('horizon')} · lookback {meta.get('lookback')} · "
            f"step {meta.get('step')} · sample_count {meta.get('sample_count')} · seed {meta.get('seed')} · CPU. "
            "Random-forest direction is its classifier (P(up) ≥ 0.5). "
            "Regressor direction is the sign of its predicted return. "
            "Kronos direction is the sign of the forecasted close change. "
            "Flat means predicting today's price."
        )
        tick = st.selectbox("Ticker", list(reports), key="vs_tick")
        rep = reports[tick]
        summary = rep["summary"]
        speed = rep["seconds_per_forecast"]
        if speed.get("mean") is not None:
            st.caption(
                f"Kronos CPU time per origin on {tick}: mean {speed['mean']:.2f}s "
                f"(min {speed['min']:.2f}s, max {speed['max']:.2f}s, n={speed['n']})."
            )
        if not summary.get("n_paired"):
            st.info("No paired origins. Lower the minimum training years, or raise history.")
            if rep.get("rf_skipped"):
                st.write(pd.DataFrame(rep["rf_skipped"]))
        else:
            k, rf, nv = summary["kronos"], summary["random_forest"], summary["naive_flat"]
            rows = pd.DataFrame({
                "Kronos": [
                    k["direction_accuracy"], None, k["mae"], k["rmse"], k["avg_pct_error"],
                    k["mean_predicted_return"],
                ],
                "Random forest": [
                    rf["direction_accuracy"], rf.get("regressor_direction_accuracy"),
                    rf["mae"], rf["rmse"], rf["avg_pct_error"], rf["mean_predicted_return"],
                ],
                "Flat price": [
                    None, None, nv["mae"], nv["rmse"], nv["avg_pct_error"], nv["mean_predicted_return"],
                ],
            }, index=[
                "Direction accuracy (forest = classifier)",
                "Regressor direction accuracy",
                "MAE ($)", "RMSE ($)", "Avg absolute % error", "Mean predicted return",
            ])
            st.dataframe(rows, width="stretch")
            st.markdown(
                f"Always-up base rate on these origins: **{_pct(k['base_rate_always_up'])}**. "
                f"Actual mean return: **{_pct(k['mean_actual_return'])}**. "
                f"Kronos was closer on **{summary['kronos_closer_count']}** origins, "
                f"the forest on **{summary['random_forest_closer_count']}**, "
                f"ties **{summary['tie_count']}** "
                f"({summary['first_origin'].date()} → {summary['last_origin'].date()}, "
                f"n={summary['n_paired']})."
            )
            if k["mae"] < rf["mae"]:
                st.success("On this window, Kronos has the lower price MAE.")
            elif rf["mae"] < k["mae"]:
                st.error("On this window, the random forest has the lower price MAE.")
            else:
                st.info("On this window, price MAE is tied.")
            kk = rep["kronos_rows"].sort_values("date")
            rr = rep["rf_rows"].sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=kk["date"], y=kk["actual_future_price"], name="Actual", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=kk["date"], y=kk["predicted_price"], name="Kronos"))
            fig.add_trace(go.Scatter(x=rr["date"], y=rr["predicted_price"], name="Random forest"))
            fig.update_layout(title=f"{tick} horizon price at each origin", height=420, yaxis_title="Price")
            st.plotly_chart(fig, width="stretch")
            show = kk.merge(rr, on="date", suffixes=("_kronos", "_rf"))
            st.dataframe(show, width="stretch")
            st.download_button(
                "Download comparison CSV", show.to_csv(index=False).encode(),
                f"{tick}_kronos_vs_rf.csv", "text/csv", key="dl_vs",
            )

with tab_pat:
    price_map = {}
    if rf_result:
        for t, r in rf_result.get("tickers", {}).items():
            if r.get("prices") is not None:
                price_map[t] = r["prices"]
    for t, fc in (st.session_state.get("kronos_forecasts") or {}).items():
        price_map.setdefault(t, fc.get("prices"))
    for t, rep in (st.session_state.get("comparison") or {}).items():
        price_map.setdefault(t, rep.get("prices"))
    if not price_map:
        choice = st.selectbox("Ticker to scan", settings["tickers"], key="pat_prefetch")
        if st.button("Load candles for the pattern scan"):
            try:
                price_map[choice] = cached_prices(choice, start_s)
                st.session_state.pattern_prices = {choice: price_map[choice]}
            except Exception as exc:
                st.error(str(exc))
        price_map = st.session_state.get("pattern_prices") or {}
    if not price_map:
        st.info("Load candles, or run a model first. The scan uses daily bars already downloaded.")
    else:
        ptick = st.selectbox("Ticker", list(price_map), key="pat_tick")
        px = price_map[ptick]
        c1, c2 = st.columns(2)
        days = c1.select_slider("Chart window (trading days)", [60, 120, 250, 500], value=120, key="pat_days")
        show_neutral = c2.toggle("Show neutral patterns (doji, spinning top)", value=False)
        dirs, hits, engine = PT.detect(px)
        conf = PT.confluence(dirs)
        st.caption(f"Pattern engine: **{engine}**. A hit on date D uses only candles up to D.")
        w = px.iloc[-int(days):]
        figp = go.Figure(go.Candlestick(
            x=w.index, open=w["Open"], high=w["High"], low=w["Low"], close=w["Close"], name=ptick,
        ))
        hw, dw = hits.loc[w.index], dirs.loc[w.index]
        for sgn, name, sym, col, ycol, off in [
            (1, "Bullish", "triangle-up", "green", "Low", 0.985),
            (-1, "Bearish", "triangle-down", "red", "High", 1.015),
            (0, "Neutral", "circle", "gray", "High", 1.03),
        ]:
            if sgn == 0 and not show_neutral:
                continue
            mask = hw & (dw == sgn)
            idx = mask.index[mask.any(axis=1)]
            if not len(idx):
                continue
            labels = [
                ", ".join(mask.columns[mask.loc[d]]) + f" | confluence {int(conf.at[d, 'score']):+d}"
                for d in idx
            ]
            figp.add_trace(go.Scatter(
                x=idx, y=w.loc[idx, ycol] * off, mode="markers", name=name,
                marker=dict(symbol=sym, size=11, color=col), text=labels,
                hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra>" + name + "</extra>",
            ))
        figp.update_layout(title=f"{ptick} candlestick patterns", height=560, xaxis_rangeslider_visible=False)
        st.plotly_chart(figp, width="stretch")
        ht = PT.hits_table(px, last_n_days=int(days))
        if not show_neutral and len(ht):
            ht = ht[ht["Direction"] != "Neutral"]
        if len(ht):
            st.dataframe(ht.style.format({"Close": "${:,.2f}"}), width="stretch", hide_index=True)
        else:
            st.info("No patterns in this window.")
        fs = PT.forward_stats(px)
        fmt = {c: "{:+.2%}" for c in fs.columns if str(c).startswith(("Avg", "Edge"))}
        fmt.update({c: "{:.0%}" for c in fs.columns if str(c).startswith("Right way")})
        st.dataframe(fs.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
        st.caption("Edge is the pattern's average forward return minus the all-days average. Small n is noise.")

with tab_help:
    st.markdown(f"""
**Random forest.** The same idea as the previous dashboard. Two forests estimate the chance the
close is higher in {H} trading days, and the percent return, which is turned back into a price.
Features are returns, moving-average ratios, volatility, volume, and range. Optional inputs are
market context (SPY, QQQ, VIX, the 10-year yield, a sector ETF), candlestick-pattern counts, and
a CSV you upload. Signals: ≥65% strong bullish, ≥55% bullish, ≤35% strong bearish, ≤45% bearish.

**Purge gap.** A training row dated D has its answer {H} trading days later. That row is used for
an origin only when the answer date is on or before the origin. Otherwise the forest would train
on prices that had not happened yet.

**Kronos.** A pretrained model that turns recent OHLCV candles into a path of future candles.
This app does not change the Kronos code. It loads **Kronos-mini** by default (about 4 million
parameters) and can switch to small or base. The device is CPU. Several one-sample forecasts are
drawn for the chart; the band is the 10th to 90th percentile of `close`. One upstream call with
`sample_count` greater than 1 averages those samples and cannot draw the band, which is why the
chart calls the model once per path.

**Head-to-head.** Both models are scored on the same origin dates. Kronos is given only the
lookback bars ending that day, and is asked for the close `horizon` trading days later. The next
origin is one step later (default: one full horizon), so the label windows do not overlap.
Kronos is **not** retrained, and its published weights may already have seen older history of
the same stocks. The forest is retrained at every origin and, when the toggle is on, also sees
market context that Kronos never sees. Read the MAE next to the flat-price MAE before deciding
that either forecast is useful.

**Candles.** The pattern tab reads the daily bars already loaded. With the sidebar toggle on,
pattern counts become random-forest features. They use only candles up to each date.

**CSV.** A value dated D is used from D forward, never backward. Use the date the number became public.

{DISCLAIMER}
""")

st.caption("Not financial advice · Yahoo Finance via yfinance · Kronos weights from Hugging Face · CPU")
