"""
CSIS 4260 - Assignment 1
Part 1: Storing and retrieving data - CSV vs Parquet benchmark.

Benchmarks the raw CSV format against Parquet with three compression
schemes (snappy, zstd, gzip) at 1x, 10x and 100x data scales.

* The 10x / 100x datasets are simulated by tiling the 1x data. They are
  written to a temporary folder and deleted afterwards, so no large
  files remain in the project (per assignment instructions).
* Writes are done in a streaming / row-group-append fashion so memory
  usage stays constant regardless of scale (the same technique works
  for both formats; CSV uses append mode, Parquet uses row groups).
* Reads are full-file loads into memory, timed for pandas and polars.

Outputs:
    benchmarks/part1_results.json   raw measurements
    benchmarks/part1_charts.png     chart used in the slides/README
"""

import json
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "all_stocks_5yr.csv")
BENCH_DIR = os.path.join(BASE_DIR, "benchmarks")
TMP_DIR = os.path.join(BENCH_DIR, "tmp_scale")
RESULTS_JSON = os.path.join(BENCH_DIR, "part1_results.json")
CHART_PNG = os.path.join(BENCH_DIR, "part1_charts.png")

SCALES = [1, 10, 100]

# Keep memory small: OHLC as float32, ticker as category.
DTYPES = {
    "date": str,
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "int64",
    "name": "category",
}

COMPRESSIONS = [
    ("snappy", "snappy", None),
    ("zstd", "zstd", None),
    ("gzip", "gzip", 1),  # level 1: much faster, still small
]


def load_base_df() -> pd.DataFrame:
    """Load the 1x dataset once (used as the tile for every scale)."""
    return pd.read_csv(DATA_PATH, dtype=DTYPES)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def bench_scale(base_df: pd.DataFrame, scale: int) -> dict:
    res = {"scale": scale, "n_rows": int(len(base_df) * scale)}
    os.makedirs(TMP_DIR, exist_ok=True)

    csv_path = os.path.join(TMP_DIR, f"data_{scale}x.csv")
    pq_paths = {
        label: os.path.join(TMP_DIR, f"data_{scale}x_{label}.parquet")
        for label, _, _ in COMPRESSIONS
    }

    # ---------------- CSV write (append mode => constant memory) --------
    def write_csv():
        base_df.to_csv(csv_path, index=False)
        for _ in range(scale - 1):
            base_df.to_csv(csv_path, index=False, header=False, mode="a")

    _, res["csv_write_s"] = timed(write_csv)
    res["csv_size_mb"] = os.path.getsize(csv_path) / 1e6

    # ---------------- Parquet writes (row-group append) ----------------
    table = pa.Table.from_pandas(base_df.reset_index(drop=True))
    for label, codec, level in COMPRESSIONS:
        def write_pq(path=pq_paths[label], codec=codec, level=level):
            kwargs = {"compression": codec}
            if level is not None:
                kwargs["compression_level"] = level
            with pq.ParquetWriter(path, table.schema, **kwargs) as writer:
                for _ in range(scale):
                    writer.write_table(table)

        _, res[f"parquet_{label}_write_s"] = timed(write_pq)
        res[f"parquet_{label}_size_mb"] = os.path.getsize(pq_paths[label]) / 1e6

    # ---------------- Reads: pandas -------------------------------------
    _, res["csv_read_pandas_s"] = timed(lambda: pd.read_csv(csv_path, dtype=DTYPES))
    for label, _, _ in COMPRESSIONS:
        _, res[f"parquet_{label}_read_pandas_s"] = timed(
            lambda p=pq_paths[label]: pd.read_parquet(p)
        )

    # ---------------- Reads: polars (extra insight) ----------------------
    _, res["csv_read_polars_s"] = timed(lambda: pl.read_csv(csv_path))
    _, res["parquet_zstd_read_polars_s"] = timed(lambda: pl.read_parquet(pq_paths["zstd"]))

    # ---------------- Cleanup -------------------------------------------
    for p in [csv_path, *pq_paths.values()]:
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(
        f"[{scale}x] csv {res['csv_size_mb']:.1f} MB | "
        f"pq_snappy {res['parquet_snappy_size_mb']:.1f} MB | "
        f"pq_zstd {res['parquet_zstd_size_mb']:.1f} MB | "
        f"pq_gzip {res['parquet_gzip_size_mb']:.1f} MB | "
        f"csv_write {res['csv_write_s']:.1f}s | pq_zstd_write {res['parquet_zstd_write_s']:.1f}s | "
        f"csv_read(pd) {res['csv_read_pandas_s']:.1f}s | pq_zstd_read(pd) {res['parquet_zstd_read_pandas_s']:.1f}s"
    )
    return res


def make_chart(results: list[dict]) -> None:
    labels = ["CSV", "Parquet\nsnappy", "Parquet\nzstd", "Parquet\ngzip"]
    scales = [r["scale"] for r in results]

    size_rows = {
        "CSV": [r["csv_size_mb"] for r in results],
        "snappy": [r["parquet_snappy_size_mb"] for r in results],
        "zstd": [r["parquet_zstd_size_mb"] for r in results],
        "gzip": [r["parquet_gzip_size_mb"] for r in results],
    }
    write_rows = {
        "CSV": [r["csv_write_s"] for r in results],
        "snappy": [r["parquet_snappy_write_s"] for r in results],
        "zstd": [r["parquet_zstd_write_s"] for r in results],
        "gzip": [r["parquet_gzip_write_s"] for r in results],
    }
    read_rows = {
        "CSV": [r["csv_read_pandas_s"] for r in results],
        "snappy": [r["parquet_snappy_read_pandas_s"] for r in results],
        "zstd": [r["parquet_zstd_read_pandas_s"] for r in results],
        "gzip": [r["parquet_gzip_read_pandas_s"] for r in results],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x = range(len(scales))
    width = 0.19

    for ax, data, title, ylabel, logy in [
        (axes[0], size_rows, "File size (MB)", "MB (log scale)", True),
        (axes[1], write_rows, "Write time (seconds)", "seconds (log scale)", True),
        (axes[2], read_rows, "Read time (seconds, pandas)", "seconds (log scale)", True),
    ]:
        for i, (key, vals) in enumerate(data.items()):
            ax.bar(
                [xi + (i - 1.5) * width for xi in x],
                vals,
                width,
                label=labels[i] if key != "CSV" else "CSV",
                color=colors[i % len(colors)],
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s}x\n({r['n_rows']:,} rows)" for s, r in zip(scales, results)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("CSV vs Parquet (snappy / zstd / gzip) at 1x, 10x and 100x scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(CHART_PNG, dpi=150)
    plt.close(fig)
    print(f"Chart saved -> {CHART_PNG}")


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    base_df = load_base_df()
    print(f"Base data: {len(base_df):,} rows, {base_df['name'].nunique()} tickers, "
          f"{base_df['date'].nunique()} trading days")

    results = [bench_scale(base_df, s) for s in SCALES]

    payload = {
        "dataset": DATA_PATH,
        "notes": (
            "10x/100x files simulated by tiling 1x data; temp files deleted after run. "
            "CSV written with pandas, Parquet written with pyarrow (row-group append). "
            "Reads measured with pandas (plus polars for reference)."
        ),
        "results": results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved -> {RESULTS_JSON}")

    make_chart(results)


if __name__ == "__main__":
    main()
