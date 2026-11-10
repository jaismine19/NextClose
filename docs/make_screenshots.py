"""
CSIS 4260 - Assignment 1
Takes screenshots of the running Streamlit dashboard using Playwright.

Prerequisites:
    1. Run part2_analysis_models.py first (creates benchmarks/*.parquet).
    2. Start the dashboard from the project root:
           .venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8501
    3. Run this script:
           .venv/bin/python docs/make_screenshots.py

Screenshots are saved to docs/screenshots/ and used by docs/build_ppt.py.
"""

import os
import sys
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 2000}, device_scale_factor=1.5
        )
        for name, query, caption in TARGETS:
            url = BASE_URL + query
            print(f"-> {url}")
            page.goto(url, wait_until="networkidle", timeout=90_000)
            # wait for the app to finish its first (cached) data load
            page.wait_for_selector("div[data-testid='stPlotlyChart']", timeout=120_000)
            page.wait_for_timeout(6000)
            path = os.path.join(OUT_DIR, f"{name}.png")
            page.screenshot(path=path, full_page=True)
            print(f"   saved {path}")
            time.sleep(1)
        browser.close()
    print("All screenshots done.")


if __name__ == "__main__":
    main()
