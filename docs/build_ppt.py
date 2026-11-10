"""
CSIS 4260 - Assignment 1
Builds docs/CSIS4260_Assignment1_Dashboard_Screenshots.pptx

Run after part1_storage_benchmark.py, part2_analysis_models.py and
docs/make_screenshots.py have produced their outputs:

    .venv/bin/python docs/build_ppt.py
"""

import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
BENCH = os.path.join(BASE, "benchmarks")
SHOTS = os.path.join(DOCS, "screenshots")

OUT_PPTX = os.path.join(DOCS, "CSIS4260_Assignment1_Dashboard_Screenshots.pptx")

DARK = RGBColor(0x1F, 0x2D, 0x3D)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GREY = RGBColor(0x58, 0x62, 0x6E)


def load_json(path):
    with open(path) as f:
        return json.load(f)


p1 = load_json(os.path.join(BENCH, "part1_results.json"))["results"]
p2t = load_json(os.path.join(BENCH, "part2_timings.json"))
p2s = load_json(os.path.join(BENCH, "part2_summary.json"))

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title=None, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.55), Inches(0.28), Inches(12.2), Inches(0.9))
        tf = tb.text_frame
        tf.text = title
        p0 = tf.paragraphs[0]
        p0.font.size = Pt(30)
        p0.font.bold = True
        p0.font.color.rgb = DARK
    if subtitle:
        tb = s.shapes.add_textbox(Inches(0.58), Inches(1.02), Inches(12.2), Inches(0.5))
        tf = tb.text_frame
        tf.text = subtitle
        tf.paragraphs[0].font.size = Pt(15)
        tf.paragraphs[0].font.color.rgb = GREY
    return s


def add_bullets(slide, items, left=0.7, top=1.7, width=12.0, height=5.3, size=16):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (text, lvl, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.bold = bold
        p.font.color.rgb = DARK if lvl == 0 else GREY
        p.space_after = Pt(8)
    return tb


def add_table(slide, rows, left, top, width, height, header=True, font=11):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    for r in range(n_rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.text = str(rows[r][c])
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(font)
            if header and r == 0:
                p.font.bold = True
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                p.font.color.rgb = DARK
    if header:
        for c in range(n_cols):
            table.cell(0, c).fill.solid()
            table.cell(0, c).fill.fore_color.rgb = ACCENT
    return table


def add_image(slide, path, left, top, width=None, height=None):
    slide.shapes.add_picture(path, Inches(left), Inches(top),
                             Inches(width) if width else None,
                             Inches(height) if height else None)


def fit_image(slide, path, left, top, max_w, max_h):
    """Scale a picture to fit inside a box and center it within that box."""
    from PIL import Image
    with Image.open(path) as im:
        w_px, h_px = im.size
    scale = min(max_w / w_px, max_h / h_px)
    w, h = w_px * scale, h_px * scale
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def fmt_s(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"


# ---------------------------------------------------------------- slide 1
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.6))
tf = tb.text_frame
tf.text = "NextClose \u2014 S&P 500 next-day close predictor"
tf.paragraphs[0].font.size = Pt(44)
tf.paragraphs[0].font.bold = True
tf.paragraphs[0].font.color.rgb = DARK
p = tf.add_paragraph()
p.text = "CSIS 4260 \u2013 Assignment 1 \u00b7 storing, benchmarking, modeling & visualizing"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p = tf.add_paragraph()
p.text = "Jaismine Kaur \u2022 screenshots, benchmarks & what I found"
p.font.size = Pt(16)
p.font.color.rgb = GREY
p = tf.add_paragraph()
p.text = "Dataset: 505 S&P 500 companies, 2013-02-08 \u2192 2018-02-07 (619,040 rows)"
p.font.size = Pt(14)
p.font.color.rgb = GREY

# ---------------------------------------------------------------- slide 2
s = add_slide("How everything is set up", "Every number in this deck can be reproduced")
add_bullets(s, [
    ("\u2022 Python 3.13 virtual environment (.venv) \u2014 dependencies pinned in requirements.txt", 0, False),
    ("   pandas 3.1, polars 2.1, pyarrow, scikit-learn, streamlit, plotly, playwright, python-pptx", 1, False),
    ("\u2022 part1_storage_benchmark.py \u2014 CSV vs Parquet (snappy/zstd/gzip) at 1x/10x/100x", 0, False),
    ("\u2022 part2_analysis_models.py \u2014 pandas vs polars, 3 indicators, 2 prediction models", 0, False),
    ("\u2022 dashboard/app.py \u2014 Streamlit dashboard (ticker search + prediction charts)", 0, False),
    ("\u2022 docs/ \u2014 the Playwright screenshot script and this deck", 0, False),
    ("\u2022 10x/100x datasets are simulated by tiling, then deleted \u2014 nothing big is ever stored", 0, False),
])

# ---------------------------------------------------------------- slide 3
s = add_slide("Part 1 \u2013 How much space does each format need?",
              "CSV vs Parquet with snappy / zstd / gzip compression")
rows = [["Scale", "Rows", "CSV (MB)", "Parquet snappy (MB)", "Parquet zstd (MB)",
         "Parquet gzip (MB)"]]
for r in p1:
    rows.append([
        f"{r['scale']}x", f"{r['n_rows']:,}",
        fmt_s(r["csv_size_mb"]),
        fmt_s(r["parquet_snappy_size_mb"]),
        fmt_s(r["parquet_zstd_size_mb"]),
        fmt_s(r["parquet_gzip_size_mb"]),
    ])
add_table(s, rows, 1.2, 1.8, 10.9, 2.0, font=13)
add_bullets(s, [
    ("\u2022 Parquet is ~3.5x smaller than CSV at every scale (29.6 MB \u2192 8.5 MB at 1x).", 0, False),
    ("\u2022 zstd and gzip save ~17% more space than snappy, at a small CPU cost.", 0, False),
    ("\u2022 At 100x this is the difference between 3.0 GB and 0.85 GB on disk.", 0, False),
], top=4.4, height=2.4)

# ---------------------------------------------------------------- slide 4
s = add_slide("Part 1 \u2013 How fast is each format? (write & read)",
              "Wall-clock seconds; reads measured with pandas (polars shown for reference)")
rows = [["Scale", "CSV write", "CSV read (pd)", "PQ zstd write", "PQ zstd read (pd)",
         "PQ zstd read (pl)"]]
for r in p1:
    rows.append([
        f"{r['scale']}x",
        fmt_s(r["csv_write_s"]),
        fmt_s(r["csv_read_pandas_s"]),
        fmt_s(r["parquet_zstd_write_s"]),
        fmt_s(r["parquet_zstd_read_pandas_s"]),
        fmt_s(r["parquet_zstd_read_polars_s"]),
    ])
add_table(s, rows, 1.2, 1.8, 10.9, 2.0, font=13)
add_bullets(s, [
    ("\u2022 At 100x: Parquet/zstd writes 9x faster (6.6s vs 60.9s) and reads ~9x faster (1.2s vs 10.3s).", 0, False),
    ("\u2022 Parquet is columnar + compressed \u2192 far less I/O, fewer bytes to parse, no type inference.", 0, False),
    ("\u2022 My pick: Parquet + zstd for analytics; CSV stays as the ingestion/source format.", 0, False),
], top=4.4, height=2.4)

# ---------------------------------------------------------------- slide 5
s = add_slide("Part 1 \u2013 The full picture")
add_image(s, os.path.join(BENCH, "part1_charts.png"), 0.9, 1.5, 11.5)

# ---------------------------------------------------------------- slide 6
s = add_slide("Part 2 \u2013 Pandas vs Polars on the data prep",
              "Same indicator formulas in both engines; results verified identical")
rows = [["Stage", "pandas (s)", "polars (s)", "Speedup"]]
for stage, pretty in [("load_csv_s", "Load CSV"), ("indicators_s", "3 indicators"),
                      ("features_s", "Lags + cleanup")]:
    pd_v = p2t[f"pandas_{stage}"]
    pl_v = p2t[f"polars_{stage}"]
    rows.append([pretty, fmt_s(pd_v), fmt_s(pl_v), f"{pd_v / max(pl_v, 1e-9):.1f}x"])
pd_total = p2t["pandas_load_csv_s"] + p2t["pandas_indicators_s"] + p2t["pandas_features_s"]
pl_total = p2t["polars_load_csv_s"] + p2t["polars_indicators_s"] + p2t["polars_features_s"]
rows.append(["Total", fmt_s(pd_total), fmt_s(pl_total), f"{pd_total / pl_total:.1f}x"])
add_table(s, rows, 1.2, 1.8, 7.0, 2.4, font=13)
add_image(s, os.path.join(BENCH, "part2_charts.png"), 8.6, 1.7, 4.2)
add_bullets(s, [
    ("\u2022 Both engines produce the same numbers (match to ~1e-4, float32 rounding).", 0, False),
    ("\u2022 Polars is ~5x faster because it vectorizes everything \u2014 no Python callbacks per group.", 0, False),
    ("\u2022 Both feed identical NumPy arrays to scikit-learn.", 0, False),
], top=4.9, height=2.2)

# ---------------------------------------------------------------- slide 7
s = add_slide("Part 2 \u2013 Indicators & models",
              "3 technical indicators, 2 prediction algorithms, 80/20 chronological split")
add_bullets(s, [
    ("Indicators I added per ticker:", 0, True),
    ("\u2022 SMA(10) \u2014 10-day simple moving average of close", 1, False),
    ("\u2022 MACD(12, 26, 9) \u2014 EMA12 \u2212 EMA26 plus 9-day signal line", 1, False),
    ("\u2022 RSI(14) \u2014 Wilder's relative strength index", 1, False),
    ("Features: open, high, low, volume, SMA10, MACD, signal, RSI, close, 1-day return \u2014 all lagged 1 day (no look-ahead)", 0, False),
    ("The models (predict next-day close for all 505 companies):", 0, True),
    ("\u2022 Linear Regression with StandardScaler \u2014 fast, extrapolates linearly", 1, False),
    ("\u2022 Random Forest (120 trees, depth 20) \u2014 catches non-linear patterns", 1, False),
    ("\u2022 Both trained on the 1-day log-return; forecast = close(t-1) \u00d7 e^r\u0302", 1, False),
    ("Backtest: first 80% of trading days train, last 20% test \u2014 chronological, so no leakage", 0, False),
])

# ---------------------------------------------------------------- slide 8
s = add_slide("Part 2 \u2013 How accurate are the models? (test set, all companies)")
overall = p2s["overall"]
rows = [["Model", "RMSE ($)", "MAE ($)", "R\u00b2"]]
for name in ["LinearRegression", "RandomForest"]:
    m = overall[name]
    rows.append([name, f"{m['rmse']:.3f}", f"{m['mae']:.3f}", f"{m['r2']:.4f}"])
add_table(s, rows, 1.2, 1.8, 6.2, 1.8, font=13)
add_bullets(s, [
    (f"Train rows: {p2s['n_train']:,} \u2022 Test rows: {p2s['n_test']:,} \u2022 Split date: {p2s['split_date']}", 0, False),
    (f"Test period: {p2s['test_date_range'][0]} \u2192 {p2s['test_date_range'][1]}", 0, False),
    ("The two models are nearly tied; per-ticker, LR wins on RMSE for 335/505 tickers, RF for 170.", 0, False),
    ("Training the log-return (not the price) keeps RF stable on high-priced stocks like PCLN (~$1,880).", 0, False),
], left=1.2, top=4.0, width=10.8, height=2.6)
fi = list(p2s["feature_importance_rf"].items())[:5]
fi_txt = ", ".join(f"{k} ({v:.2f})" for k, v in fi)
s2 = add_slide("Part 2 \u2013 What the Random Forest actually learned", "")
add_bullets(s2, [
    ("Top-5 features:", 0, True),
    (fi_txt, 1, False),
    ("\u2022 Yesterday's return, volume and the momentum indicators do the work.", 1, False),
    ("\u2022 close(t-1) adds little beyond ret1 \u2014 classic near-random-walk prices.", 1, False),
    ("\u2022 Every ticker's RMSE / MAE / R\u00b2 is saved and shown in the dashboard.", 1, False),
])

# ---------------------------------------------------------------- slide 9
s = add_slide("Part 3 \u2013 Picking a dashboard library",
              "Streamlit vs Dash vs Reflex \u2014 and why I chose Streamlit")
add_table(s, [
    ["Library", "Language model", "Learning curve", "Interactive charts", "Deployment"],
    ["Streamlit", "Python script, top-down reruns", "Very low", "Plotly native", "Easy (Community Cloud)"],
    ["Dash (Plotly)", "Callbacks / React under the hood", "Medium", "Plotly native", "Easy"],
    ["Reflex", "Compiles Python to React", "Steeper", "Custom components", "Younger ecosystem"],
], 0.9, 1.7, 11.5, 2.2, font=12)
add_bullets(s, [
    ("Why Streamlit won me over:", 0, True),
    ("\u2022 Fastest way to build a ticker-search + charts app; caching via @st.cache_data.", 1, False),
    ("\u2022 Plotly works out of the box \u2192 candlesticks, zoom, hover tooltips.", 1, False),
    ("\u2022 Free one-command hosting, so the grader can verify it easily.", 1, False),
    ("What's inside: candlestick + SMA/MACD/RSI panels, MAPE & directional accuracy,", 0, False),
    ("rolling RMSE, feature importance, zoomable backtest, CSV download.", 0, False),
    ("Run it: .venv/bin/streamlit run dashboard/app.py", 0, False),
], top=4.3, height=2.8)

# ---------------------------------------------------------------- slides 10-15
shots = [
    ("dashboard_home.png", "Dashboard \u2013 landing view", [
        "Landing view with the default ticker",
        "Hero stats: last close, 52-week range, volume",
        "Backtest metric cards (RMSE / MAE / MAPE / R\u00b2 / direction)",
        "Candlestick with SMA(10) / SMA(30) and volume",
    ]),
    ("aapl_rf.png", "AAPL \u2013 hero, metrics & candlestick", [
        "Selected via search + ticker dropdown",
        "Per-model performance cards",
        "Next-day forecast with % move vs last close",
        "Candlestick chart with 1M / 6M / 1Y zoom buttons",
    ]),
    ("aapl_backtest.png", "AAPL \u2013 backtest & error analysis", [
        "Actual vs predicted close (test period)",
        "Rolling 20-day RMSE per model",
        "Random Forest feature importance (top 10)",
        "Backtest zoom slider in the sidebar",
    ]),
    ("aapl_diagnostics.png", "AAPL \u2013 diagnostics & downloads", [
        "Predicted vs actual scatter with perfect-fit line",
        "Histogram of prediction errors",
        "One-click CSV download of predictions",
        "Recent rows with features & predictions",
    ]),
    ("msft_rf.png", "MSFT \u2013 Random Forest model", [
        "Same layout for every one of the 505 tickers",
        "Random Forest selected",
    ]),
    ("googl_lr.png", "GOOGL \u2013 Linear Regression model", [
        "Same layout for every one of the 505 tickers",
        "Linear Regression selected",
    ]),
]
for fname, caption, bullets in shots:
    path = os.path.join(SHOTS, fname)
    if os.path.exists(path):
        s = add_slide(caption)
        add_bullets(s, [("\u2022 " + b, 0, False) for b in bullets],
                    left=0.8, top=2.0, width=5.9, height=4.6, size=16)
        fit_image(s, path, left=6.9, top=1.8, max_w=5.9, max_h=5.5)

prs.save(OUT_PPTX)
print(f"Saved {OUT_PPTX}")
