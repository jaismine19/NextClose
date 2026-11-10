"""
CSIS 4260 - Assignment 1
Part 3: Visual dashboard for the price-prediction models.

Run from the project root with:
    .venv/bin/streamlit run dashboard/app.py

Features:
  * ticker search across all 505 S&P 500 companies
  * hero header with last close, day change, 52-week range, avg volume
  * per-model metric cards: RMSE, MAE, MAPE, R^2, directional accuracy
  * next-day price forecast from both models
  * candlestick chart with SMA(10)/SMA(30) and volume, plus MACD/RSI panels
  * backtest charts with a zoomable date window and rolling 20-day RMSE
  * predicted-vs-actual diagnostics and RF feature importance
  * one-click download of the predictions

Deep links (used for automated screenshots):
    /?ticker=AAPL&model=Random%20Forest
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="S&P 500 Price Prediction Dashboard",
    page_icon="\U0001F4C8",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "all_stocks_5yr.csv")
PRED_PATH = os.path.join(BASE_DIR, "benchmarks", "predictions.parquet")
METRICS_PATH = os.path.join(BASE_DIR, "benchmarks", "ticker_metrics.parquet")
SUMMARY_PATH = os.path.join(BASE_DIR, "benchmarks", "part2_summary.json")

MODEL_COL = {"Linear Regression": "pred_lr", "Random Forest": "pred_rf"}
MODEL_COLOR = {"Linear Regression": "#E45756", "Random Forest": "#54A24B"}
MODEL_LABEL = {"Linear Regression": "LR", "Random Forest": "RF"}

# ----------------------------------------------------------------------
# Cached data loading (polars: much faster to load than pandas)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tickers() -> list:
    return (
        pl.read_csv(CSV_PATH, columns=["name"])
        .get_column("name")
        .unique()
        .sort()
        .to_list()
    )


@st.cache_data(show_spinner=False)
def load_predictions() -> pl.DataFrame:
    return pl.read_parquet(PRED_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> pl.DataFrame:
    return pl.read_parquet(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_model_summary() -> dict:
    with open(SUMMARY_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_ticker_history(ticker: str) -> pl.DataFrame:
    """Raw OHLCV for one ticker plus the technical indicators from Part 2."""
    df = pl.read_csv(CSV_PATH).filter(pl.col("name") == ticker).sort("date")
    df = (
        df.with_columns(
            pl.col("close").rolling_mean(10).over("name").alias("sma10"),
            pl.col("close").rolling_mean(30).over("name").alias("sma30"),
            (
                pl.col("close").ewm_mean(span=12, adjust=False).over("name")
                - pl.col("close").ewm_mean(span=26, adjust=False).over("name")
            ).alias("macd"),
        )
        .with_columns(
            pl.col("macd").ewm_mean(span=9, adjust=False).over("name").alias("macd_signal"),
        )
        .with_columns(
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
            pl.col("close").diff().over("name").alias("delta"),
        )
        .with_columns(
            pl.when(pl.col("delta").is_null()).then(None)
            .when(pl.col("delta") > 0).then(pl.col("delta"))
            .otherwise(0.0).alias("gain"),
            pl.when(pl.col("delta").is_null()).then(None)
            .when(pl.col("delta") < 0).then(-pl.col("delta"))
            .otherwise(0.0).alias("loss"),
        )
        .with_columns(
            (
                100
                - 100
                / (
                    1
                    + pl.col("gain").ewm_mean(alpha=1 / 14, adjust=False, min_samples=14).over("name")
                    / pl.col("loss").ewm_mean(alpha=1 / 14, adjust=False, min_samples=14).over("name")
                )
            )
            .fill_null(100.0)
            .alias("rsi"),
            pl.int_range(pl.len()).over("name").alias("_r"),
        )
        .with_columns(
            pl.when(pl.col("_r") >= 13).then(pl.col("rsi")).otherwise(None).alias("rsi"),
        )
        .drop(["delta", "gain", "loss", "_r"])
    )
    return df


# ----------------------------------------------------------------------
# Extra backtest metrics (computed from the saved predictions)
# ----------------------------------------------------------------------
def ticker_extra_metrics(preds: pl.DataFrame) -> dict:
    p = preds.sort("date").with_columns(
        ((pl.col("pred_lr") - pl.col("close")).abs() / pl.col("close") * 100).alias("ape_lr"),
        ((pl.col("pred_rf") - pl.col("close")).abs() / pl.col("close") * 100).alias("ape_rf"),
        pl.col("close").shift(1).alias("close_lag1"),
    ).drop_nulls("close_lag1")
    out = {"mape_lr": p["ape_lr"].mean(), "mape_rf": p["ape_rf"].mean()}
    for m, col in MODEL_COL.items():
        short = MODEL_LABEL[m]
        up = ((p[col] - p["close_lag1"]) > 0) & ((p["close"] - p["close_lag1"]) > 0)
        down = ((p[col] - p["close_lag1"]) < 0) & ((p["close"] - p["close_lag1"]) < 0)
        out[f"dir_{short}"] = (up | down).sum() / p.height
    return out


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def price_chart(df: pl.DataFrame, ticker: str) -> go.Figure:
    d = df.to_pandas()
    up_color = np.where(d["close"].to_numpy() >= d["open"].to_numpy(), "#26a69a", "#ef5350")
    fig = make_subplots(
        rows=2, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} \u2014 daily OHLC with SMA(10) / SMA(30)", "Volume"),
    )
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["sma10"], name="SMA(10)",
                             line=dict(color="#ff7f0e", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["sma30"], name="SMA(30)",
                             line=dict(color="#9467bd", width=1.4)), row=1, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="Volume",
                         marker_color=up_color, opacity=0.55), row=2, col=1)
    fig.update_layout(
        height=640,
        title=dict(
            text=f"{ticker} \u2014 Price history: daily candlesticks with SMA(10) / SMA(30)",
            x=0.01, font=dict(size=16),
        ),
        legend=dict(orientation="h", y=1.02), margin=dict(t=80, b=20),
        hovermode="x unified", xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (shares)", row=2, col=1)
    fig.update_xaxes(rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#f0f4f8",
    ), row=2, col=1)
    return fig


def momentum_chart(df: pl.DataFrame, ticker: str) -> go.Figure:
    d = df.to_pandas()
    fig = make_subplots(
        rows=2, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
        subplot_titles=(f"{ticker} \u2014 MACD (12, 26, 9)", f"{ticker} \u2014 RSI (14)"),
    )
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd"], name="MACD",
                             line=dict(color="#2ca02c", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], name="Signal",
                             line=dict(color="#d62728", width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], name="Histogram",
                         marker_color="#bdbdbd", opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["rsi"], name="RSI(14)",
                             line=dict(color="#1f77b4", width=1.4)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#c44e52", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#55a868", row=2, col=1)
    fig.update_layout(
        height=520,
        title=dict(
            text=f"{ticker} \u2014 Momentum indicators: MACD (12, 26, 9) and RSI (14)",
            x=0.01, font=dict(size=16),
        ),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=80, b=20), hovermode="x unified",
    )
    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    return fig


def backtest_chart(preds: pl.DataFrame, ticker: str, models: list) -> go.Figure:
    p = preds.sort("date").to_pandas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["date"], y=p["close"], name="Actual close",
                             line=dict(color="#1f77b4", width=1.8)))
    for m in models:
        fig.add_trace(go.Scatter(
            x=p["date"], y=p[MODEL_COL[m]], name=f"Predicted \u2014 {m}",
            line=dict(color=MODEL_COLOR[m], width=1.8, dash="dot"),
        ))
    title = f"{ticker} \u2014 Backtest: actual vs predicted"
    if len(models) == 1:
        title += f" ({models[0]})"
    else:
        title += f" ({' & '.join(models)})"
    fig.update_layout(
        height=440,
        title=dict(text=title, x=0.01, font=dict(size=16)),
        hovermode="x unified", margin=dict(t=70, b=20),
        legend=dict(orientation="h", y=1.06),
    )
    fig.update_yaxes(title_text="Price ($)")
    return fig


def rolling_rmse_chart(preds: pl.DataFrame, ticker: str, models: list) -> go.Figure:
    p = preds.sort("date").to_pandas()
    fig = go.Figure()
    for m in models:
        err2 = (p[MODEL_COL[m]] - p["close"]) ** 2
        roll = np.sqrt(err2.rolling(20, min_periods=1).mean())
        fig.add_trace(go.Scatter(x=p["date"], y=roll, name=f"{m} (20-day RMSE)",
                                 line=dict(color=MODEL_COLOR[m], width=1.6)))
    fig.update_layout(
        height=440,
        title=dict(
            text=f"{ticker} \u2014 Rolling 20-day RMSE on the test period",
            x=0.01, font=dict(size=16),
        ),
        hovermode="x unified", margin=dict(t=70, b=20),
        legend=dict(orientation="h", y=1.06),
    )
    fig.update_yaxes(title_text="RMSE ($)")
    return fig


def scatter_residual(preds: pl.DataFrame, ticker: str, model: str) -> go.Figure:
    p = preds.to_pandas()
    col = MODEL_COL[model]
    err = p[col] - p["close"]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted vs actual", "Prediction errors"),
    )
    fig.add_trace(go.Scatter(x=p["close"], y=p[col], mode="markers",
                             marker=dict(size=4, opacity=0.4, color=MODEL_COLOR[model]),
                             name="points"), row=1, col=1)
    lo, hi = float(p["close"].min()), float(p["close"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="black", dash="dash"), name="perfect"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=err, nbinsx=60, marker_color=MODEL_COLOR[model],
                               name="errors"), row=1, col=2)
    fig.update_layout(
        height=380,
        title=dict(
            text=f"{ticker} \u2014 {model}: predicted vs actual and error distribution",
            x=0.01, font=dict(size=16),
        ),
        showlegend=False, margin=dict(t=60, b=20),
    )
    fig.update_xaxes(title_text="Actual close", row=1, col=1)
    fig.update_yaxes(title_text="Predicted close", row=1, col=1)
    fig.update_xaxes(title_text="Error (predicted - actual)", row=1, col=2)
    return fig


def feature_importance_chart(summary: dict) -> go.Figure:
    fi = list(summary["feature_importance_rf"].items())[:10][::-1]
    fig = go.Figure(go.Bar(
        x=[v for _, v in fi], y=[k for k, _ in fi], orientation="h",
        marker_color="#2e86c1",
    ))
    fig.update_layout(
        height=380,
        title=dict(
            text="Random Forest feature importance \u2014 top 10 (all companies)",
            x=0.01, font=dict(size=16),
        ),
        margin=dict(t=60, b=20, l=10), xaxis_title="Mean decrease in impurity",
    )
    return fig


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem;}
    .hero {
        background: linear-gradient(90deg, #1f2d3d 0%, #2e86c1 100%);
        padding: 1.5rem 1.8rem; border-radius: 14px; color: #ffffff;
        margin-bottom: 0.8rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 700;}
    .hero p {margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fb; border: 1px solid #e6eaf0;
        border-radius: 10px; padding: 10px 12px;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

params = st.query_params
param_ticker = params.get("ticker")
param_model = params.get("model")
if isinstance(param_ticker, list):
    param_ticker = param_ticker[0] if param_ticker else None
if isinstance(param_model, list):
    param_model = param_model[0] if param_model else None

tickers = load_tickers()
predictions = load_predictions()
metrics = load_metrics()
summary = load_model_summary()

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("\U0001F4C8 S&P 500 Predictor")
    st.caption("CSIS 4260 \u2014 Assignment 1 \u2014 Part 3 \u00b7 Jaismine Kaur")
    st.markdown("---")
    search = st.text_input("\U0001F50D Search ticker", value="", placeholder="e.g. AAPL")
    filtered = [t for t in tickers if search.upper() in t] or tickers
    default_ticker = param_ticker if param_ticker in filtered else filtered[0]
    default_idx = filtered.index(default_ticker) if default_ticker in filtered else 0
    ticker = st.selectbox("Company ticker", filtered, index=default_idx)

    st.markdown("---")
    radio_options = ["Random Forest", "Linear Regression", "Both"]
    model = st.radio(
        "Prediction model",
        radio_options,
        index=radio_options.index(param_model) if param_model in radio_options else 0,
    )

    st.markdown("---")
    test_dates = predictions.filter(pl.col("name") == ticker).get_column("date").unique().sort()
    t0 = dt.date.fromisoformat(test_dates[0])
    t1 = dt.date.fromisoformat(test_dates[-1])
    zoom = st.slider(
        "Backtest zoom window",
        min_value=t0, max_value=t1, value=(t0, t1), format="YYYY-MM-DD",
        help="Narrow the test period shown in the backtest charts.",
    )
    zoom_lo, zoom_hi = zoom[0].isoformat(), zoom[1].isoformat()

    st.markdown("---")
    st.markdown(
        "**The data**: 505 S&P 500 companies, 2013-02-08 to 2018-02-07 "
        "(619,040 rows). Models are backtested on an 80/20 chronological split."
    )
    st.markdown(
        "**The models** (trained on every company):\n"
        "- Linear Regression (scaled features)\n"
        "- Random Forest (120 trees)"
    )
    st.caption("The full benchmarks and reasoning live in README.md")

# ---------------- hero header ----------------
history = load_ticker_history(ticker)
ticker_preds = predictions.filter(pl.col("name") == ticker).sort("date")
ticker_metrics = metrics.filter(pl.col("name") == ticker).row(0, named=True)
extra = ticker_extra_metrics(ticker_preds)

dates = history["date"].to_list()
closes = history["close"].to_list()
last_close, prev_close = closes[-1], closes[-2]
day_change = (last_close - prev_close) / prev_close * 100
year_hist = history.tail(252)
lo52 = float(year_hist["low"].min())
hi52 = float(year_hist["high"].max())
avg_vol = float(year_hist["volume"].mean())

st.markdown(
    f"""
    <div class="hero">
        <h1>{ticker} \u2014 Price Prediction Dashboard</h1>
        <p>Daily OHLCV from {dates[0]} to {dates[-1]} \u00b7 {history.height:,} trading days \u00b7
        backtested on the last 20% of trading days \u00b7 built with Streamlit + Plotly</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_cols = st.columns(5)
stat_cols[0].metric("Last close", f"${last_close:,.2f}",
                    delta=f"{day_change:+.2f}% vs previous day")
stat_cols[1].metric("52-week low", f"${lo52:,.2f}")
stat_cols[2].metric("52-week high", f"${hi52:,.2f}")
stat_cols[3].metric("Avg volume (1Y)", f"{avg_vol / 1e6:,.1f}M")
stat_cols[4].metric("Test days", f"{ticker_preds.height:,}",
                    help="Number of backtest days for this ticker")

st.markdown("---")

# ---------------- model metric cards ----------------
models_shown = ["Random Forest", "Linear Regression"] if model == "Both" else [model]
st.subheader("\U0001F3AF Backtest performance (last 20% of trading days)")
for m in models_shown:
    short = MODEL_LABEL[m]
    if m == "Random Forest":
        rmse, mae, r2, next_pred = (
            ticker_metrics["rmse_rf"], ticker_metrics["mae_rf"],
            ticker_metrics["r2_rf"], ticker_metrics["next_pred_rf"],
        )
        mape, dir_acc = extra["mape_rf"], extra["dir_RF"]
        colour = "rgba(84,162,75,0.12)"
    else:
        rmse, mae, r2, next_pred = (
            ticker_metrics["rmse_lr"], ticker_metrics["mae_lr"],
            ticker_metrics["r2_lr"], ticker_metrics["next_pred_lr"],
        )
        mape, dir_acc = extra["mape_lr"], extra["dir_LR"]
        colour = "rgba(228,87,86,0.12)"
    st.markdown(
        f"<div style='background:{colour};padding:8px 14px;border-radius:10px;"
        f"margin-top:0.6rem;'><b>{m}</b></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    cols[0].metric("RMSE", f"${rmse:,.2f}")
    cols[1].metric("MAE", f"${mae:,.2f}")
    cols[2].metric("MAPE", f"{mape:,.2f}%")
    cols[3].metric("R\u00b2", f"{r2:,.3f}")
    cols[4].metric("Directional accuracy", f"{dir_acc * 100:,.1f}%",
                   help="Share of days the model got the move direction right")
    cols[5].metric(
        f"Next-day forecast ({ticker_metrics['last_date']} + 1d)",
        f"${next_pred:,.2f}",
        delta=f"{((next_pred / last_close) - 1) * 100:+.2f}% vs last close",
    )

st.markdown("---")

# ---------------- price & momentum ----------------
st.subheader("\U0001F4CA Price history & technical indicators")
st.plotly_chart(price_chart(history, ticker), width="stretch")
st.plotly_chart(momentum_chart(history, ticker), width="stretch")

# ---------------- backtest charts ----------------
zoom_preds = ticker_preds.filter(
    (pl.col("date") >= zoom_lo) & (pl.col("date") <= zoom_hi)
)
st.subheader("\U0001F4C8 Model backtest (80/20 chronological split)")
st.plotly_chart(backtest_chart(zoom_preds, ticker, models_shown), width="stretch")

b1, b2 = st.columns(2)
with b1:
    st.plotly_chart(rolling_rmse_chart(zoom_preds, ticker, models_shown),
                    width="stretch")
with b2:
    st.plotly_chart(feature_importance_chart(summary), width="stretch")

# ---------------- diagnostics ----------------
st.subheader("\U0001F50D Diagnostics")
if model == "Both":
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_residual(ticker_preds, ticker, "Random Forest"),
                        width="stretch")
    with c2:
        st.plotly_chart(scatter_residual(ticker_preds, ticker, "Linear Regression"),
                        width="stretch")
else:
    st.plotly_chart(scatter_residual(ticker_preds, ticker, model),
                    width="stretch")

# ---------------- download + raw data ----------------
dl_col, _ = st.columns([1, 3])
with dl_col:
    st.download_button(
        "\u2B07\uFE0F Download predictions (.csv)",
        data=ticker_preds.to_pandas().to_csv(index=False).encode(),
        file_name=f"{ticker}_predictions.csv",
        mime="text/csv",
    )

with st.expander("\U0001F5C2 Recent rows with features & predictions"):
    feat_rows = history.select(["date", "open", "high", "low", "close", "volume"]
                               + ["sma10", "sma30", "macd", "rsi"]).tail(15)
    merged = feat_rows.join(
        ticker_preds.select(["date", "pred_lr", "pred_rf"]), on="date", how="left"
    ).sort("date", descending=True)
    st.dataframe(merged.to_pandas().round(3), width="stretch", hide_index=True)
