# CSIS 4260 — Assignment 1
## Storing, Analyzing and Visualizing S&P 500 Stock Data

**Repository:** https://github.com/jaismine19/CSIS4260_Assignment1 (public — read
access is open to everyone; this is the submission link for saleemn@douglascollege.ca)

Student: **Jaismine Kaur**

This assignment walked me through the whole pipeline of working with stock data:
deciding how to store it, how to analyze it quickly, how to build a couple of
prediction models on it, and finally how to wrap all of that into a dashboard that
a person can actually use. Below is what I did for each part, why I did it that
way, and how to run everything yourself.

The dataset is `data/all_stocks_5yr.csv` — daily open/high/low/close/volume for
**505 S&P 500 companies** from 2013-02-08 to 2018-02-07, which works out to
**619,040 rows** and about 29 MB.

| Part | What it covers | Code | What it produces |
|------|----------------|------|------------------|
| 1 | Storing & retrieving data (CSV vs Parquet) | `part1_storage_benchmark.py` | `benchmarks/part1_results.json`, `benchmarks/part1_charts.png` |
| 2 | Pandas vs Polars, indicators, prediction models | `part2_analysis_models.py` | `benchmarks/predictions.parquet`, `ticker_metrics.parquet`, `part2_*.json` |
| 3 | Visual dashboard | `dashboard/app.py` | Streamlit app (screenshots in `docs/` + the PPTX deck) |

---

## Setup (virtual environment)

Everything runs in a Python 3.13 virtual environment. From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is the file the venv tooling generated for me (`pip freeze`),
so it pins every dependency and the grader can reproduce my exact environment.

### Part 1 — storage benchmark
```bash
.venv/bin/python part1_storage_benchmark.py
```
> A note on the 10x and 100x datasets: since I'm not allowed to include those
> files, the script **simulates them by tiling the 1x data**, writes them to a
> temporary folder, and **deletes them automatically** when it's done. Nothing
> large is ever left behind or committed.

### Part 2 — analysis + models
```bash
.venv/bin/python part2_analysis_models.py
```
This generates the predictions and per-ticker metrics that the dashboard reads.

### Part 3 — dashboard
```bash
.venv/bin/streamlit run dashboard/app.py
```
Then open http://localhost:8501. Search any ticker in the sidebar (e.g. `AAPL`).
You can also deep-link straight to a ticker and model:
`/?ticker=AAPL&model=Random%20Forest`.

### Regenerating the screenshots / slides
```bash
.venv/bin/python docs/make_screenshots.py   # needs the dashboard running on :8501
.venv/bin/python docs/build_ppt.py          # rebuilds the PPTX deck
```

---

## Part 1 — Storing and retrieving data: CSV vs Parquet

### The thinking

The assignment asks whether to keep the data as the CSV it came in or convert it
to Parquet, and to back that up with benchmarks at 1x, 10x and 100x scale.

My understanding going in was:

* **CSV** is row-oriented, uncompressed text. Every read has to parse every byte
  of every line, and string values have to be type-inferred on the way in. It's
  human-readable, which is nice, but nothing about it is optimized for machines.
* **Parquet** (Apache Arrow / `pyarrow`) is **column-oriented and binary**, with
  per-column encoding and built-in compression. Files are much smaller, a schema
  is stored inside the file (so no guessing types at read time), and you only
  decode the columns you actually ask for.

I tested three compression codecs, since the assignment asked about compression
schemes:

| Codec | Why I tested it |
|-------|-----------------|
| `snappy` | Parquet's default — extremely fast, decent ratio |
| `zstd`  | usually the best size/speed balance, fast decompression |
| `gzip`  | slightly better ratio, slower; I used level 1 to stay realistic |

### How I measured

For each scale (1x = 619,040 rows, 10x = 6,190,400, 100x = 61,904,000) I measured:

1. **Write time** — CSV written with pandas (append mode), Parquet written with
   `pyarrow.parquet.ParquetWriter` appending row groups. Both approaches keep
   memory constant, so neither side gets an unfair memory advantage.
2. **File size** on disk.
3. **Read time** — a full load into memory with pandas, plus polars reads as an
   extra data point (it sets up Part 2 nicely).

### What I found

MacBook Pro (Apple silicon, 16 GB RAM, 10 cores), Python 3.13, pandas 3.1 /
pyarrow / polars 2.1:

| Scale | Rows | Format | Size (MB) | Write (s) | Read (s, pandas) | Read (s, polars) |
|-------|------|--------|----------:|----------:|-----------------:|-----------------:|
| 1x    | 619,040   | CSV            | 29.6  | 0.58  | 0.096  | 0.014 |
|       |           | Parquet snappy | 10.3  | 0.059 | 0.0061 | — |
|       |           | Parquet zstd   | **8.5** | **0.064** | **0.0067** | 0.016 |
|       |           | Parquet gzip   | 8.6   | 0.15  | 0.014  | — |
| 10x   | 6,190,400  | CSV            | 295.8 | 5.98  | 1.06   | 0.18 |
|       |           | Parquet snappy | 102.6 | 0.60  | 0.061  | — |
|       |           | Parquet zstd   | **85.0** | 0.68  | **0.042** | 0.074 |
|       |           | Parquet gzip   | 86.3  | 2.21  | 0.077  | — |
| 100x  | 61,904,000 | CSV            | 2,958.1 | 60.9 | 10.3  | 2.24 |
|       |           | Parquet snappy | 1,026.5 | 5.87 | 1.29 | — |
|       |           | Parquet zstd   | **849.7** | 6.62 | 1.16 | **0.69** |
|       |           | Parquet gzip   | 863.5  | 15.7 | 0.96  | — |

Charts: `benchmarks/part1_charts.png`.

### My recommendation

* **At 1x**, honestly, CSV is fine — everything is sub-second and you can open it
  in any editor. There's no *need* to convert, but Parquet/zstd costs nothing and
  already saves 3.5x on disk.
* **At 10x** the gap stops being cosmetic: Parquet writes ~9x faster and reads
  15–25x faster.
* **At 100x** it's basically a decision made for you. Writing the CSV takes about
  61 seconds versus ~6.6 seconds for Parquet/zstd; reading it back with pandas
  takes ~10 seconds versus ~1.2 seconds (polars gets the Parquet read down to
  ~0.7s); and the file shrinks from ~3.0 GB to ~0.85 GB.

So my choice: **store the analytics data as Parquet with zstd compression**.
zstd gave me the best size/speed balance in my runs — snappy writes a touch
faster, but zstd is smaller and reads just as fast. The original CSV stays around
as the ingestion/source format only. Bonus point: as the data grows, Parquet lets
you read individual columns and push filters down to the file level, which CSV
simply cannot do.

---

## Part 2 — Pandas vs Polars, indicators, prediction models

### Why compare these two libraries?

* **Pandas** is the library everyone learns first. It's NumPy-backed, but a lot
  of per-group work falls back to Python-level loops — the classic example is
  `groupby(...).transform(lambda ...)`, which runs a Python callback once per
  group (505 times here).
* **Polars** is a Rust-based engine with a lazy query optimizer, vectorized
  kernels and real multithreading. Expressions like
  `pl.col("close").rolling_mean(10).over("name")` compile into one optimized
  plan instead of 505 Python callbacks.

On 619k rows both libraries feel fast, but the whole point of this assignment is
that the data will grow 10x and 100x, and the benchmark below shows which one
scales better.

### The indicators I added

| Indicator | Formula | Why it's useful |
|-----------|---------|-----------------|
| SMA(10) | 10-day simple moving average of close | shows trend direction |
| MACD(12,26,9) | EMA12 − EMA26, plus 9-day signal line + histogram | catches momentum shifts |
| RSI(14) | Wilder's relative strength index of daily gains/losses | flags overbought/oversold |

I also added a **1-day return** column. One design decision worth calling out:
because the models predict the *next* day's close, I lagged **every** feature by
one day — the model only ever sees information that existed at the close of day
t−1. No look-ahead leakage. I implemented the indicators identically in pandas
and polars and verified the two engines agree to ~1e-4 (which is just float32
rounding noise).

### Pandas vs Polars timings (619,040 rows)

| Stage | pandas (s) | polars (s) | Speedup |
|-------|-----------:|-----------:|--------:|
| Load CSV | 0.103 | 0.018 | 5.7x |
| 3 indicators (SMA/MACD/RSI) | 0.143 | 0.026 | 5.4x |
| Lag features + cleanup | 0.031 | 0.010 | 3.1x |
| **Total data prep** | **0.277** | **0.054** | **5.1x** |

Chart: `benchmarks/part2_charts.png`. The NumPy matrices I fed to scikit-learn
were verified to be identical between the two engines. Model training itself
doesn't care which engine prepared the data: Linear Regression took 0.04 s,
Random Forest took 67 s.

### The models

I trained two algorithms to predict the next-day closing price for all 505
companies, both on the **one-day log-return** (standard practice for prices —
returns are stationary and comparable across tickers). The dollar forecast is
recovered as `close(t-1) × exp(ŕ)`:

1. **Linear Regression** with `StandardScaler` — simple, fast, and it extrapolates
   gracefully, which turned out to matter (see below).
2. **Random Forest** (120 trees, `max_depth=20`, `min_samples_leaf=5`) — captures
   non-linear relationships between the lagged prices and indicators.

The split is **chronological**: sorted by date, first 80% of trading days train
the model, last 20% backtest it. No shuffling, so there's no way for future
information to leak into training. Features (all lagged one day): `open_l1,
high_l1, low_l1, volume_l1, sma10_l1, macd_l1, macd_signal_l1, rsi_l1, close_l1,
ret1_l1`.

### How they did (test period, all companies)

| Model | Overall RMSE ($) | Overall MAE ($) | Overall R² | Median per-ticker RMSE ($) | Tickers where it wins (lower RMSE) |
|-------|-----------------:|----------------:|-----------:|---------------------------:|-----------------------------------:|
| Linear Regression | **2.241** | **0.908** | 0.99969 | 0.950 | 335 / 505 |
| Random Forest     | 2.260 | 0.912 | 0.99969 | 0.951 | 170 / 505 |

Metrics are in dollars after back-transforming the log-return predictions.
Train rows: 486,870 (through 2017-02-10). Test rows: 125,093 (2017-02-13 →
2018-02-07).

**Feature importance (Random Forest, top 5):** `ret1_l1` 0.18, `volume_l1` 0.15,
`rsi_l1` 0.14, `macd_signal_l1` 0.14, `macd_l1` 0.13. Yesterday's return, volume
and the momentum indicators carry the signal — which fits the near-random-walk
behaviour of daily prices.

### A mistake I caught (and fixed) along the way

My first version of the model used the *same day's* open/high/low as features —
which made the "next-day" prediction trivially easy (the close is almost always
inside the day's range). That's look-ahead leakage, so I rebuilt it with
everything lagged one day.

The second issue was more interesting. When I trained on **dollar prices**
directly, Random Forest looked terrible overall (RMSE ~11.6 vs ~2.2 for LR) even
though it was fine on most tickers. The culprit was Priceline (PCLN, ~$1,880):
its test-period price ran about 20% above anything seen in training, and tree
leaves can only predict the average of the training prices in a leaf — they can't
extrapolate. Linear Regression just kept extrapolating linearly. Switching the
target to the **log-return** fixed it: PCLN's RF RMSE went from $255 to $27.

### Part 2 verdict

* **Polars is the better engine for this workload.** Identical results ~5x faster,
  with expressions that are honestly easier to read than pandas lambdas. And the
  gap only widens at 10x/100x.
* **The two models end up nearly tied** at ~0.9997 R² — next-day close is
  overwhelmingly yesterday's close. Linear Regression wins outright on 335/505
  tickers, trains in 0.04 s versus 67 s, and handles out-of-range prices safely.
* **Recommendation:** scaled Linear Regression with a log-return target as the
  primary model for this dataset. Random Forest is a solid secondary model and
  would pull ahead if I added non-linear features like sector or sentiment data.

---

## Part 3 — Visual dashboard

### Dashboard library research

| Library | How it works | What I liked | What I didn't |
|---------|--------------|--------------|---------------|
| **Streamlit** | plain Python script, top-down reruns | fastest to build, `@st.cache_data` caching, native Plotly, free hosting | limited fine-grained interactivity |
| **Dash** | Flask + React callbacks | full Plotly ecosystem, production-grade | callbacks add boilerplate; steeper learning curve |
| **Reflex** | Python compiled to React | very flexible components | younger ecosystem, more setup |

I went with **Streamlit**. The assignment needs a ticker-searchable page with a
handful of linked charts — Streamlit does that in the least code, its caching
handles the 30 MB CSV nicely, and the grader can run and verify it with one
command. Dash and Reflex are great tools, but they'd add ceremony without adding
value for this particular dashboard.

### What the dashboard shows

* **Hero header** with the ticker, date range, last close + day change, 52-week
  low/high and average volume.
* Sidebar: **ticker search + selection** (all 505 companies are available), a
  model choice (Random Forest / Linear Regression / both), and a
  **zoom window slider** for the backtest charts.
* Metric cards per model: **RMSE, MAE, MAPE, R², directional accuracy** (how often
  the model got the move direction right) and the **next-day forecast**.
* Charts: a **candlestick** chart with SMA(10)/SMA(30) and volume (with 1M/6M/1Y
  range buttons), **MACD and RSI(14)** panels, the backtest window (actual vs
  predicted), a **rolling 20-day RMSE**, a predicted-vs-actual scatter with error
  histogram, and the **Random Forest feature importance**.
* A one-click **CSV download** of the predictions and a table of recent rows.

Screenshots of the running app are in `docs/screenshots/` and assembled into
`docs/CSIS4260_Assignment1_Dashboard_Screenshots.pptx`.

---

## Files

```
CSIS4260_Assignment1/
├── README.md                          ← this file
├── requirements.txt                   ← generated by the venv (pip freeze)
├── part1_storage_benchmark.py         ← Part 1
├── part2_analysis_models.py           ← Part 2
├── dashboard/app.py                   ← Part 3 (Streamlit)
├── docs/
│   ├── make_screenshots.py            ← Playwright screenshot generator
│   ├── build_ppt.py                   ← PPTX builder
│   ├── screenshots/                   ← dashboard screenshots
│   └── CSIS4260_Assignment1_Dashboard_Screenshots.pptx
├── benchmarks/                        ← outputs (JSON metrics, parquet, charts)
└── data/all_stocks_5yr.csv            ← the provided dataset (1x only; 10x/100x are never stored)
```
