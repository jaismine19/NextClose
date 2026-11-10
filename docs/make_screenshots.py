"""
CSIS 4260 - Assignment 1
Takes screenshots of the running Streamlit dashboard using Playwright.

Prerequisites:
    1. Run part2_analysis_models.py first (creates benchmarks/*.parquet).
    2. Start the dashboard from the project root:
           .venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8501
    3. Run this script:
           .venv/bin/python docs/make_screenshots.py

For each target we save a viewport screenshot (used in the slide deck) and,
for the deep-dive ticker, additional screenshots of the backtest and
diagnostics sections.

Output: docs/screenshots/*.png (used by docs/build_ppt.py)
"""

import os
import time

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

TARGETS = [
    ("dashboard_home", "", "Landing view (default ticker)"),
    ("aapl_rf", "?ticker=AAPL&model=Random%20Forest", "AAPL - Random Forest"),
    ("msft_rf", "?ticker=MSFT&model=Random%20Forest", "MSFT - Random Forest"),
    ("googl_lr", "?ticker=GOOGL&model=Linear%20Regression", "GOOGL - Linear Regression"),
]

# extra section shots for the deep-dive ticker (AAPL)
SECTION_SHOTS = [
    ("aapl_backtest", "Model backtest", "AAPL - backtest section"),
    ("aapl_diagnostics", "Diagnostics", "AAPL - diagnostics section"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 1700}, device_scale_factor=1.5
        )
        for name, query, caption in TARGETS:
            url = BASE_URL + query
            print(f"-> {url}")
            page.goto(url, wait_until="networkidle", timeout=90_000)
            page.wait_for_selector("div[data-testid='stPlotlyChart']", timeout=120_000)
            page.wait_for_timeout(6000)
            page.screenshot(path=os.path.join(OUT_DIR, f"{name}.png"))
            print(f"   saved {name}.png")
            time.sleep(1)

        # section shots for AAPL
        page.goto(BASE_URL + "?ticker=AAPL&model=Random%20Forest",
                  wait_until="networkidle", timeout=90_000)
        page.wait_for_selector("div[data-testid='stPlotlyChart']", timeout=120_000)
        page.wait_for_timeout(6000)
        for name, section_text, caption in SECTION_SHOTS:
            page.get_by_text(section_text, exact=False).first.scroll_into_view_if_needed()
            page.wait_for_timeout(2500)
            page.screenshot(path=os.path.join(OUT_DIR, f"{name}.png"))
            print(f"   saved {name}.png")
            time.sleep(1)

        browser.close()
    print("All screenshots done.")


if __name__ == "__main__":
    main()
