"""
CSIS 4260 - Assignment 1
Part 2: Manipulating, analyzing data and building models.

1. Compares Pandas vs Polars for: CSV loading, technical-indicator
   computation, feature engineering and model data preparation.
2. Enhances the dataset with three technical indicators:
     - SMA(10)   simple moving average of close
     - MACD(12,26,9) + signal line
     - RSI(14)   Wilder's relative strength index
   plus two engineered features (1-day lag close, 1-day return).
3. Trains two models to predict the next-day closing price for every
   company, using an 80/20 chronological (backtesting) split:
     - Linear Regression (with StandardScaler)
     - Random Forest Regressor
4. Saves everything the dashboard (Part 3) needs.

Outputs (benchmarks/):
    predictions.parquet      test-set actual vs predictions for every ticker
    ticker_metrics.parquet   per-ticker RMSE / MAE / R^2 + next-day prediction
    part2_timings.json       pandas vs polars timings + indicator verification
    part2_summary.json       model info, overall metrics, feature importance
    part2_charts.png         pandas vs polars timing chart for the slides
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "all_stocks_5yr.csv")
BENCH_DIR = os.path.join(BASE_DIR, "benchmarks")

FEATURES = [
    "open", "high", "low", "volume",
    "sma10", "macd", "macd_signal", "rsi",
    "close_lag1", "ret1",
]
TARGET = "close"

PD_DTYPES = {
    "date": str, "open": "float32", "high": "float32",
    "low": "float32", "close": "float32", "volume": "int64",
    "name": "category",
}
PL_SCHEMA = {
    "date": pl.String, "open": pl.Float32, "high": pl.Float32,
    "low": pl.Float32, "close": pl.Float32, "volume": pl.Int64,
    "name": pl.String,
}

timings = {}


def tic(label):
    timings["_t"] = time.perf_counter()
    timings["_label"] = label


def toc():
    timings[timings["_label"]] = time.perf_counter() - timings["_t"]


# ----------------------------------------------------------------------
# Pandas implementation
# ----------------------------------------------------------------------
def load_pandas():
    tic("pandas_load_csv_s")
    df = pd.read_csv(DATA_PATH, dtype=PD_DTYPES)
    toc()
    return df.sort_values(["name", "date"], kind="stable").reset_index(drop=True)


def add_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    tic("pandas_indicators_s")
    g = df.groupby("name", sort=False)

    df["sma10"] = g["close"].transform(lambda s: s.rolling(10).mean())

    e12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    e26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = e12 - e26
    df["macd_signal"] = df.groupby("name", sort=False)["macd"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    df["_gain"] = gain
    df["_loss"] = loss
    avg_gain = df.groupby("name", sort=False)["_gain"].transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    avg_loss = df.groupby("name", sort=False)["_loss"].transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    df["rsi"] = 100 - 100 / (1 + avg_gain / avg_loss)
    # flat periods (no gains AND no losses) -> RSI convention of 100
    df["rsi"] = df["rsi"].fillna(100.0)
    # restore the 13-row warm-up window per ticker
    df["_r"] = g.cumcount()
    df["rsi"] = df["rsi"].where(df["_r"] >= 13)

    df = df.drop(columns=["_gain", "_loss", "_r"])
    toc()
    return df


def add_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    tic("pandas_features_s")
    g = df.groupby("name", sort=False)
    df["close_lag1"] = g["close"].shift(1)
    df["ret1"] = g["close"].pct_change()
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    toc()
    return df


# ----------------------------------------------------------------------
# Polars implementation (identical indicator definitions)
# ----------------------------------------------------------------------
def load_polars():
    tic("polars_load_csv_s")
    df = pl.read_csv(DATA_PATH, schema_overrides=PL_SCHEMA)
    toc()
    return df.sort(["name", "date"])


def add_indicators_polars(df: pl.DataFrame) -> pl.DataFrame:
    tic("polars_indicators_s")
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
    toc()
    return df


def add_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    tic("polars_features_s")
    df = df.with_columns(
        pl.col("close").shift(1).over("name").alias("close_lag1"),
        pl.col("close").pct_change().over("name").alias("ret1"),
    ).drop_nulls(subset=FEATURES + [TARGET])
    toc()
    return df


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def verify(pd_df: pd.DataFrame, pl_df: pl.DataFrame) -> dict:
    cols = ["sma10", "macd", "macd_signal", "macd_hist", "rsi", "close_lag1", "ret1"]
    pl_pd = pl_df.select(["name", "date"] + cols).to_pandas()
    merged = pd_df[["name", "date"] + cols].merge(
        pl_pd, on=["name", "date"], suffixes=("_pd", "_pl")
    )
    diffs = {}
    for c in cols:
        d = (merged[f"{c}_pd"].astype(float) - merged[f"{c}_pl"].astype(float)).abs()
        diffs[c] = float(d.max())
    return diffs


# ----------------------------------------------------------------------
# Modeling (80/20 chronological split, shared by both engines' output)
# ----------------------------------------------------------------------
def build_models(df: pd.DataFrame):
    tic("split_and_prep_s")
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    X_train = train[FEATURES].to_numpy(dtype=np.float64)
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    X_test = test[FEATURES].to_numpy(dtype=np.float64)
    y_test = test[TARGET].to_numpy(dtype=np.float64)
    toc()

    # ----- Linear Regression (scaled) -----
    tic("train_lr_s")
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    lr.fit(X_train, y_train)
    toc()

    # ----- Random Forest -----
    tic("train_rf_s")
    rf = RandomForestRegressor(
        n_estimators=120, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    toc()

    tic("predict_s")
    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)
    toc()

    def overall_metrics(y, p):
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
        }

    model_summary = {
        "overall": {
            "LinearRegression": overall_metrics(y_test, pred_lr),
            "RandomForest": overall_metrics(y_test, pred_rf),
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_date_range": [str(train["date"].min()), str(train["date"].max())],
        "test_date_range": [str(test["date"].min()), str(test["date"].max())],
        "split_date": str(split_date),
        "features": FEATURES,
        "feature_importance_rf": dict(
            sorted(zip(FEATURES, rf.feature_importances_.tolist()), key=lambda t: -t[1])
        ),
        "indicators": {
            "SMA10": "10-day simple moving average of close",
            "MACD": "EMA12 - EMA26 with 9-day signal line",
            "RSI14": "Wilder relative strength index (14-day)",
        },
        "models": {
            "LinearRegression": "Ridge-free OLS with StandardScaler",
            "RandomForest": "120 trees, max_depth=20, min_samples_leaf=5",
        },
    }

    # ----- per-ticker test metrics + next-day prediction -----
    tic("per_ticker_s")
    test_preds = test[["date", "name", TARGET]].copy()
    test_preds["pred_lr"] = pred_lr
    test_preds["pred_rf"] = pred_rf

    rows = []
    last_rows = df.groupby("name", sort=False).tail(1)
    X_last = last_rows[FEATURES].to_numpy(dtype=np.float64)
    last_pred_lr = lr.predict(X_last)
    last_pred_rf = rf.predict(X_last)
    last_map_lr = dict(zip(last_rows["name"], last_pred_lr))
    last_map_rf = dict(zip(last_rows["name"], last_pred_rf))

    for name, grp in test_preds.groupby("name", sort=False):
        y = grp[TARGET].to_numpy()
        rows.append({
            "name": name,
            "n_test": int(len(grp)),
            "rmse_lr": float(np.sqrt(mean_squared_error(y, grp["pred_lr"].to_numpy()))),
            "mae_lr": float(mean_absolute_error(y, grp["pred_lr"].to_numpy())),
            "r2_lr": float(r2_score(y, grp["pred_lr"].to_numpy())),
            "rmse_rf": float(np.sqrt(mean_squared_error(y, grp["pred_rf"].to_numpy()))),
            "mae_rf": float(mean_absolute_error(y, grp["pred_rf"].to_numpy())),
            "r2_rf": float(r2_score(y, grp["pred_rf"].to_numpy())),
            "last_date": str(grp["date"].max()),
            "last_close": float(grp[TARGET].max()),
            "next_pred_lr": float(last_map_lr[name]),
            "next_pred_rf": float(last_map_rf[name]),
        })
    metrics_df = pd.DataFrame(rows)
    toc()

    return test_preds, metrics_df, model_summary


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_outputs(test_preds: pd.DataFrame, metrics_df: pd.DataFrame, summary: dict):
    pl.from_pandas(test_preds).write_parquet(os.path.join(BENCH_DIR, "predictions.parquet"))
    pl.from_pandas(metrics_df).write_parquet(os.path.join(BENCH_DIR, "ticker_metrics.parquet"))

    with open(os.path.join(BENCH_DIR, "part2_timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    with open(os.path.join(BENCH_DIR, "part2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved predictions.parquet, ticker_metrics.parquet, part2_timings.json, part2_summary.json")


def make_timing_chart():
    stages = [
        "load_csv_s", "indicators_s", "features_s",
    ]
    pretty = {
        "load_csv_s": "Load CSV",
        "indicators_s": "3 indicators",
        "features_s": "Lags + cleanup",
    }
    pd_vals = [timings[f"pandas_{s}"] for s in stages]
    pl_vals = [timings[f"polars_{s}"] for s in stages]

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, pd_vals, 0.36, label="pandas", color="#4C72B0")
    ax.bar(x + 0.18, pl_vals, 0.36, label="polars", color="#DD8452")
    for xi, v in zip(x - 0.18, pd_vals):
        ax.text(xi, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + 0.18, pl_vals):
        ax.text(xi, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[s] for s in stages])
    ax.set_ylabel("seconds")
    ax.set_title("Pandas vs Polars - data prep stages (619,040 rows)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BENCH_DIR, "part2_charts.png"), dpi=150)
    plt.close(fig)
    print("Saved part2_charts.png")


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)

    print("=" * 70)
    print("Loading + indicators + features: pandas vs polars")
    print("=" * 70)

    pd_df = load_pandas()
    pd_df = add_indicators_pandas(pd_df)
    pd_df = add_features_pandas(pd_df)

    pl_df = load_polars()
    pl_df = add_indicators_polars(pl_df)
    pl_df = add_features_polars(pl_df)

    diffs = verify(pd_df, pl_df)
    print("\nIndicator verification (max |pandas - polars|):")
    for k, v in diffs.items():
        print(f"  {k:>12}: {v:.2e}")

    print("\npandas timings:")
    for k, v in timings.items():
        if k.startswith("pandas"):
            print(f"  {k:<28} {v:.3f}s")
    print("polars timings:")
    for k, v in timings.items():
        if k.startswith("polars"):
            print(f"  {k:<28} {v:.3f}s")

    print("\n" + "=" * 70)
    print("Modeling (80/20 chronological split)")
    print("=" * 70)
    test_preds, metrics_df, summary = build_models(pd_df)
    print(json.dumps(summary["overall"], indent=2))
    print(f"train rows={summary['n_train']:,}  test rows={summary['n_test']:,}")
    print(f"split date={summary['split_date']}")
    print(f"test period={summary['test_date_range']}")

    save_outputs(test_preds, metrics_df, summary)
    make_timing_chart()

    print("\nDone. Per-ticker metrics (first 10):")
    print(metrics_df[["name", "n_test", "rmse_lr", "rmse_rf", "r2_lr", "r2_rf"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
