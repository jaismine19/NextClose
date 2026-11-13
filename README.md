# NextClose — predicting the next day's closing price for 505 S&P 500 companies

**CSIS 4260 — Assignment 1** · Student: **Jaismine Kaur**

This assignment took me through the whole journey of working with stock data:
deciding how to store it, how to analyze it quickly, how to build two prediction
models, and finally how to put everything into a dashboard that people can
actually use. Below is what I did for each part, why I did it that way, and how
you can run everything yourself.

The dataset is `data/all_stocks_5yr.csv`. It holds daily prices (open, high, low,
close) and volume for **505 S&P 500 companies**, from 2013-02-08 to 2018-02-07.
That is **619,040 rows** in total, about 29 MB.

| Part | What it covers | Code | What it produces |
|------|----------------|------|------------------|
| 1 | Storing & retrieving data (CSV vs Parquet) | `part1_storage_benchmark.py` | `benchmarks/part1_results.json`, `benchmarks/part1_charts.png` |
| 2 | Pandas vs Polars, indicators, prediction models | `part2_analysis_models.py` | `benchmarks/predictions.parquet`, `ticker_metrics.parquet`, `part2_*.json` |
| 3 | Visual dashboard | `dashboard/app.py` | Streamlit app (screenshots in `docs/` + the PPTX deck) |

---

## Setup (virtual environment)

Everything runs in a Python 3.13 virtual environment. From the project folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is the dependency file my venv tool generated for me
(`pip freeze`). It pins every package, so the grader can recreate my exact setup.

### Part 1 — storage benchmark
```bash
.venv/bin/python part1_storage_benchmark.py
```
> A note on the 10x and 100x datasets: I am not allowed to include those files,
> so the script **pretends the data grew by copying the 1x data 10 or 100 times**.
> It writes the big files to a temporary folder and **deletes them automatically**
> when it finishes. Nothing large is ever left behind or committed.

### Part 2 — analysis + models
```bash
.venv/bin/python part2_analysis_models.py
```
This creates the predictions and per-company metrics that the dashboard reads.

### Part 3 — dashboard
```bash
.venv/bin/streamlit run dashboard/app.py
```
Then open http://localhost:8501. Search any ticker in the sidebar (for example
`AAPL`). You can also jump straight to a ticker and model with a link:
`/?ticker=AAPL&model=Random%20Forest`.

The finished dashboard screenshots and the slide deck are included in `docs/`.

---

## Part 1 — Storing and retrieving data: CSV vs Parquet

### The thinking

The assignment asks: should the data stay as a CSV, or should I convert it to
Parquet? And it asks me to prove the answer with benchmarks at 1x, 10x and 100x
scale.

Here is what I knew going in:

* **CSV** is plain text, one line per row, no compression. Every time you read it,
  the program has to read every single byte and guess whether each column is text
  or a number. It is easy for humans to read, but nothing about it is built for
  speed or size.
* **Parquet** (Apache Arrow / `pyarrow`) stores data **column by column** instead
  of row by row, in a compact binary form with built-in compression. Files are
  much smaller, the column types are saved inside the file (no guessing at read
  time), and you can read just the columns you need.

I also tested three compression options, since the assignment asked about
compression:

| Codec | Why I tested it |
|-------|-----------------|
| `snappy` | Parquet's default — very fast, decent size savings |
| `zstd`  | usually the best balance of speed and size |
| `gzip`  | a bit smaller, but slower; I used level 1 to keep it realistic |

### How I measured

For each scale (1x = 619,040 rows, 10x = 6,190,400, 100x = 61,904,000) I measured:

1. **Write time** — saving the file.
2. **File size** on disk.
3. **Read time** — loading the whole file back into memory with pandas (plus a
   polars read as a bonus data point, which sets up Part 2).

Both formats were written in a way that keeps memory constant, so neither side
got an unfair advantage.

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

* **At 1x**, to be honest, CSV is fine. Everything takes less than a second and
  you can open the file in any editor. There is no *need* to convert, but
  Parquet/zstd costs nothing and already saves 3.5x on disk.
* **At 10x** the difference stops being cosmetic. Parquet writes about 9x faster
  and reads 15–25x faster.
* **At 100x** the choice makes itself. Writing the CSV takes about 61 seconds;
  Parquet/zstd takes about 6.6 seconds. Reading the CSV with pandas takes about
  10 seconds; Parquet takes about 1.2 seconds (polars reads Parquet in ~0.7s).
  And the file shrinks from ~3.0 GB to ~0.85 GB.

So my choice: **keep the data as Parquet with zstd compression**. In my runs zstd
gave the best size/speed balance — snappy writes a touch faster, but zstd is
smaller and reads just as fast. The original CSV stays around as the source file
only. One more bonus: as the data grows, Parquet can read only the columns it
needs and skip parts of the file, which CSV simply cannot do.

---

## Part 2 — Pandas vs Polars, indicators, prediction models

### Why compare these two libraries?

* **Pandas** is the library everyone learns first. It is built on NumPy, but many
  group-by-group tasks fall back to running a Python function 505 times (once per
  company). That gets slow.
* **Polars** is a Rust-based engine. It plans the whole calculation up front and
  runs it on multiple cores at once, working on whole columns in one go. One
  expression like `pl.col("close").rolling_mean(10).over("name")` replaces
  505 Python function calls.

On 619k rows both feel fast. But this assignment assumes the data will grow 10x
and 100x, and the benchmark below shows which one handles that better.

### The indicators I added

| Indicator | Formula | Why it's useful |
|-----------|---------|-----------------|
| SMA(10) | 10-day simple moving average of close | shows the trend direction |
| MACD(12,26,9) | EMA12 − EMA26, plus a 9-day signal line + histogram | catches momentum shifts |
| RSI(14) | Wilder's relative strength index of daily gains/losses | flags overbought/oversold |

I also added a **1-day return** column. One design decision worth mentioning:
because the models predict the *next* day's close, I shifted **every** feature by
one day. The model only sees information that existed at the end of day t−1.
No cheating (no look-ahead). I wrote the indicators the same way in pandas and
polars and checked that both give the same numbers (they agree to ~1e-4, which
is just rounding noise).

### Pandas vs Polars timings (619,040 rows)

| Stage | pandas (s) | polars (s) | Speedup |
|-------|-----------:|-----------:|--------:|
| Load CSV | 0.103 | 0.018 | 5.7x |
| 3 indicators (SMA/MACD/RSI) | 0.143 | 0.026 | 5.4x |
| Lag features + cleanup | 0.031 | 0.010 | 3.1x |
| **Total data prep** | **0.277** | **0.054** | **5.1x** |

Chart: `benchmarks/part2_charts.png`. The data I fed to scikit-learn was
verified to be identical from both engines. Training itself does not care which
engine prepared the data: Linear Regression took 0.04 s, Random Forest took 67 s.

### The models

I trained two models to predict the next-day closing price for all 505 companies.
Both predict the **daily percent change** instead of the dollar price (the usual
way to do this — percent changes behave the same across companies, dollar prices
do not). The dollar forecast is then just yesterday's close plus that change.

1. **Linear Regression** (with `StandardScaler`) — simple, fast, and it handles
   prices outside the range it has seen before, which turned out to matter (see
   the story below).
2. **Random Forest** (120 trees, `max_depth=20`, `min_samples_leaf=5`) — catches
   non-linear patterns between the lagged prices and indicators.

The split is **chronological**: I sorted by date and used the first 80% of
trading days to train, the last 20% to test (backtesting). No shuffling, so no
future information can sneak into training. Features (all shifted one day):
`open_l1, high_l1, low_l1, volume_l1, sma10_l1, macd_l1, macd_signal_l1, rsi_l1,
close_l1, ret1_l1`.

### How they did (test period, all companies)

| Model | Overall RMSE ($) | Overall MAE ($) | Overall R² | Median per-ticker RMSE ($) | Tickers where it wins (lower RMSE) |
|-------|-----------------:|----------------:|-----------:|---------------------------:|-----------------------------------:|
| Linear Regression | **2.241** | **0.908** | 0.99969 | 0.950 | 335 / 505 |
| Random Forest     | 2.260 | 0.912 | 0.99969 | 0.951 | 170 / 505 |

RMSE = average error in dollars (bigger = worse). MAE = average absolute error.
R² = how much of the price movement the model explains (1.0 would be perfect).
Train rows: 486,870 (through 2017-02-10). Test rows: 125,093 (2017-02-13 →
2018-02-07).

**Feature importance (Random Forest, top 5):** `ret1_l1` 0.18, `volume_l1` 0.15,
`rsi_l1` 0.14, `macd_signal_l1` 0.14, `macd_l1` 0.13. Yesterday's return, volume
and the momentum indicators carry the signal — which makes sense, because daily
prices are almost a random walk.

### A mistake I caught (and fixed) along the way

My first version used the *same day's* open/high/low as features. That made the
"next-day" prediction trivially easy, because the close is almost always inside
the day's range. It was look-ahead leakage (cheating), so I rebuilt it with every
feature shifted one day back.

The second problem was more interesting. When I trained on **dollar prices**
directly, Random Forest looked terrible overall (RMSE ~11.6 vs ~2.2 for LR), even
though it was fine on most companies. The culprit was Priceline (PCLN, ~$1,880):
its test-period price ran about 20% above anything seen in training. A decision
tree can only guess the average price of its training examples — it cannot guess
beyond them. Linear Regression, on the other hand, just kept drawing the straight
line. Switching to the **percent change** fixed it: PCLN's RF error dropped from
$255 to $27.

### Part 2 verdict

* **Polars is the better engine here.** Same results, about 5x faster, and the
  expressions are honestly easier to read than pandas lambdas. The gap only gets
  bigger at 10x/100x.
* **The two models end up nearly tied** at R² ≈ 0.9997 — tomorrow's close is
  mostly today's close. Linear Regression wins outright on 335/505 companies,
  trains in 0.04 s instead of 67 s, and handles out-of-range prices safely.
* **Recommendation:** scaled Linear Regression on the percent change as the main
  model for this dataset. Random Forest is a good second model and would pull
  ahead if I added non-linear features like sector or news sentiment data.

---

## Part 3 — Visual dashboard

### Dashboard library research

| Library | How it works | What I liked | What I didn't |
|---------|--------------|--------------|---------------|
| **Streamlit** | plain Python script, reruns top to bottom | fastest to build, built-in caching, native Plotly, free hosting | limited fine-grained interactivity |
| **Dash** | Flask + React callbacks | full Plotly ecosystem, production-grade | callbacks add boilerplate; steeper learning curve |
| **Reflex** | Python compiled to React | very flexible components | younger ecosystem, more setup |

I went with **Streamlit**. The assignment needs a page with ticker search and a
handful of linked charts — Streamlit does that in the least code, its caching
handles the 30 MB CSV nicely, and the grader can run and check it with one
command. Dash and Reflex are great tools, but they would add extra setup without
adding value for this particular dashboard.

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
│   ├── screenshots/                   ← dashboard screenshots
│   └── CSIS4260_Assignment1_Dashboard_Screenshots.pptx
├── benchmarks/                        ← outputs (JSON metrics, parquet, charts)
└── data/all_stocks_5yr.csv            ← the provided dataset (1x only; 10x/100x are never stored)
```
