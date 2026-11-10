"""
CSIS 4260 - Assignment 1
Part 3: Visual dashboard for the price-prediction models.

Run from the project root with:
    .venv/bin/streamlit run dashboard/app.py

The dashboard lets the user search/select any S&P 500 ticker and shows:
  * full price history with SMA(10), MACD and RSI(14) technical panels
  * 80/20 backtest window: actual vs predicted close for both models
  * per-ticker error metrics (RMSE, MAE, R^2) and the next-day forecast

Deep links (used for automated screenshots):
    /?ticker=AAPL&model=Random%20Forest
"""

import os

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

FEATURES = ["open", "high", "low", "volume", "sma10", "macd", "macd_signal", "rsi",
            "close_lag1", "ret1"]

MODEL_COL = {"Linear Regression": "pred_lr", "Random Forest": "pred_rf"}
MODEL_COLOR = {"Linear Regression": "#E45756", "Random Forest": "#54A24B"}


# ----------------------------------------------------------------------
# Cached data loading (polars is used because it is much faster to load)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tickers() -> list[str]:
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
def load_ticker_history(ticker: str) -> pl.DataFrame:
    """Raw OHLCV for one ticker plus the same technical indicators as Part 2."""
    df = pl.read_csv(CSV_PATH).filter(pl.col("name") == ticker).sort("date")
    df = (
        df.with_columns(
            pl.col("close").rolling_mean(10).over("name").alias("sma10"),
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
# Charts
# ----------------------------------------------------------------------
def technical_chart(df: pl.DataFrame, ticker: str) -> go.Figure:
    d = df.to_pandas()
    fig = make_subplots(
        rows=4, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.42, 0.18, 0.2, 0.2],
        subplot_titles=(
            f"{ticker} - close & SMA(10)", "Volume", "MACD (12, 26, 9)", "RSI (14)"
        ),
    )
    fig.add_trace(go.Scatter(x=d["date"], y=d["close"], name="Close",
                             line=dict(color="#1f77b4", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["sma10"], name="SMA(10)",
                             line=dict(color="#ff7f0e", width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="Volume",
                         marker_color="#9ecae1"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd"], name="MACD",
                             line=dict(color="#2ca02c", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], name="Signal",
                             line=dict(color="#d62728", width=1.2)), row=3, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], name="Histogram",
                         marker_color="#bdbdbd"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["rsi"], name="RSI(14)",
                             line=dict(color="#9467bd", width=1.2)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#c44e52", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#55a868", row=4, col=1)
    fig.update_layout(
        height=820, legend=dict(orientation="h", y=1.02),
        margin=dict(t=60, b=20), hovermode="x unified",
    )
    return fig


def backtest_chart(preds: pl.DataFrame, ticker: str, model: str) -> go.Figure:
    p = preds.filter(pl.col("name") == ticker).sort("date").to_pandas()
    col = MODEL_COL[model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["date"], y=p["close"], name="Actual close",
                             line=dict(color="#1f77b4", width=1.8)))
    fig.add_trace(go.Scatter(x=p["date"], y=p[col], name=f"Predicted ({model})",
                             line=dict(color=MODEL_COLOR[model], width=1.8, dash="dot")))
    fig.update_layout(
        height=430, title=f"Backtest (test period): actual vs predicted - {model}",
        hovermode="x unified", margin=dict(t=60, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def scatter_residual(preds: pl.DataFrame, ticker: str, model: str) -> go.Figure:
    p = preds.filter(pl.col("name") == ticker).to_pandas()
    col = MODEL_COL[model]
    err = p[col] - p["close"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted vs actual", "Prediction errors"))
    fig.add_trace(go.Scatter(x=p["close"], y=p[col], mode="markers",
                             marker=dict(size=4, opacity=0.4, color=MODEL_COLOR[model]),
                             name="points"), row=1, col=1)
    lo, hi = float(p["close"].min()), float(p["close"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="black", dash="dash"), name="perfect"), row=1, col=1)
    fig.add_trace(go.Histogram(x=err, nbinsx=60, marker_color=MODEL_COLOR[model],
                               name="errors"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, margin=dict(t=50, b=20))
    fig.update_xaxes(title_text="Actual close", row=1, col=1)
    fig.update_yaxes(title_text="Predicted close", row=1, col=1)
    fig.update_xaxes(title_text="Error (predicted - actual)", row=1, col=2)
    return fig


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
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

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("\U0001F4C8 S&P 500 Predictor")
    st.caption("CSIS 4260 - Assignment 1 - Part 3")
    st.markdown("---")
    search = st.text_input("\U0001F50D Search ticker", value="", placeholder="e.g. AAPL")
    filtered = [t for t in tickers if search.upper() in t]
    if not filtered:
        filtered = tickers
    default_ticker = param_ticker if param_ticker in filtered else filtered[0]
    default_idx = filtered.index(default_ticker) if default_ticker in filtered else 0
    ticker = st.selectbox("Company ticker", filtered, index=default_idx)

    st.markdown("---")
    model = st.radio(
        "Prediction model",
        ["Random Forest", "Linear Regression", "Both"],
        index=0 if param_model not in ["Linear Regression", "Both"] else
              (["Linear Regression", "Both"].index(param_model)),
    )
    st.markdown("---")
    st.markdown(
        "**Data**: 505 S&P 500 companies, 2013-02-08 to 2018-02-07 "
        "(619,040 rows). 80/20 chronological split for backtesting."
    )
    st.markdown(
        "**Models** (trained on all companies):\n"
        "- Linear Regression (scaled features)\n"
        "- Random Forest (120 trees)"
    )
    st.caption("Benchmarks & analysis: see README.md")

# ---------------- data for selected ticker ----------------
history = load_ticker_history(ticker)
ticker_preds = predictions.filter(pl.col("name") == ticker).sort("date")
ticker_metrics = metrics.filter(pl.col("name") == ticker).row(0, named=True)

st.title(f"{ticker} \u2014 Price Prediction Dashboard")
st.caption(
    f"Daily data from {history['date'][0]} to {history['date'][-1]} "
    f"({history.height:,} trading days). "
    f"Backtest period: {ticker_preds['date'][0]} \u2192 {ticker_preds['date'][-1]} "
    f"({ticker_preds.height:,} days)."
)

# ---------------- metric cards ----------------
models_shown = (
    ["Random Forest", "Linear Regression"] if model == "Both" else [model]
)
cols = st.columns(len(models_shown) * 4 if len(models_shown) == 2 else 4)
idx = 0
for m in models_shown:
    if m == "Random Forest":
        rmse, mae, r2, next_pred = (
            ticker_metrics["rmse_rf"], ticker_metrics["mae_rf"],
            ticker_metrics["r2_rf"], ticker_metrics["next_pred_rf"],
        )
        colour = "rgba(84,162,75,0.12)"
    else:
        rmse, mae, r2, next_pred = (
            ticker_metrics["rmse_lr"], ticker_metrics["mae_lr"],
            ticker_metrics["r2_lr"], ticker_metrics["next_pred_lr"],
        )
        colour = "rgba(228,87,86,0.12)"
    with cols[idx]:
        st.markdown(
            f"<div style='background:{colour};padding:10px 14px;border-radius:10px;'>"
            f"<b style='font-size:0.85em'>{m}</b></div>", unsafe_allow_html=True)
        st.metric("RMSE (test)", f"${rmse:,.2f}")
    with cols[idx + 1]:
        st.metric("MAE (test)", f"${mae:,.2f}")
    with cols[idx + 2]:
        st.metric("R\u00b2 (test)", f"{r2:,.3f}")
    with cols[idx + 3]:
        st.metric(
            f"Next-day forecast ({ticker_metrics['last_date']} + 1d)",
            f"${next_pred:,.2f}",
            delta=f"vs last close ${ticker_metrics['last_close']:,.2f}",
        )
    idx += 4

st.markdown("---")

# ---------------- technical chart ----------------
st.subheader("\U0001F4CA Price history & technical indicators")
st.plotly_chart(technical_chart(history, ticker), use_container_width=True)

# ---------------- backtest charts ----------------
st.subheader("\U0001F3AF Model backtest (80/20 split)")
if model == "Both":
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(backtest_chart(ticker_preds, ticker, "Random Forest"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(backtest_chart(ticker_preds, ticker, "Linear Regression"),
                        use_container_width=True)
else:
    st.plotly_chart(backtest_chart(ticker_preds, ticker, model), use_container_width=True)

if model == "Both":
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(scatter_residual(ticker_preds, ticker, "Random Forest"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(scatter_residual(ticker_preds, ticker, "Linear Regression"),
                        use_container_width=True)
else:
    st.plotly_chart(scatter_residual(ticker_preds, ticker, model), use_container_width=True)

# ---------------- raw data table ----------------
with st.expander("\U0001F5C2 Recent rows with features & predictions"):
    feat_rows = history.select(["date", "open", "high", "low", "close", "volume"]
                               + ["sma10", "macd", "rsi"]).tail(15)
    merged = feat_rows.join(
        ticker_preds.select(["date", "pred_lr", "pred_rf"]), on="date", how="left"
    ).sort("date", descending=True)
    st.dataframe(merged.to_pandas().round(3), use_container_width=True, hide_index=True)
