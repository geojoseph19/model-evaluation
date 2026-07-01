#!/usr/bin/env python3
"""Generate PDF and/or PNG report from an existing ALD evaluation run.

Usage:
    python3 export.py                        # export runs/latest
    python3 export.py runs/2026-07-01_v3     # export a specific run
    python3 export.py --format pdf           # PDF only
    python3 export.py --format png           # PNG only
    python3 export.py --format pdf png       # both (default)

Requires:
    pip install playwright
    playwright install chromium
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_report(run_dir: Path, formats: list[str]) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error(
            "Playwright not installed.\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
        return

    report_html = Path(__file__).parent / "index.html"
    if not report_html.exists():
        logger.error(f"index.html not found at {report_html}")
        return

    report_data_file = run_dir / "report_data.json"
    if not report_data_file.exists():
        logger.error(f"report_data.json not found in {run_dir}")
        return

    report_data = report_data_file.read_text(encoding="utf-8")
    logger.info(f"Run:     {run_dir}")
    logger.info(f"Formats: {', '.join(formats)}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Inject data before any page script runs — bypasses fetch entirely
        # (fetch is blocked on file:// URLs in Chromium)
        page.add_init_script(f"""
            window.__REPORT_DATA__ = {report_data};
            localStorage.setItem('theme', 'light');
        """)
        page.goto(f"file://{report_html.resolve()}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)  # let Chart.js finish rendering

        # Predictions table is interactive-only — skip in exports
        page.evaluate("document.getElementById('section-predictions').style.display = 'none'")

        if "pdf" in formats:
            pdf_path = run_dir / "report.pdf"
            page.pdf(path=str(pdf_path), format="A3", landscape=True, print_background=True)
            logger.info(f"  PDF → {pdf_path}")

        if "png" in formats:
            png_path = run_dir / "report.png"
            page.screenshot(path=str(png_path), full_page=True)
            logger.info(f"  PNG → {png_path}")

        browser.close()


def main() -> None:
    runs_root = Path(__file__).parent / "runs"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to a run directory (default: runs/latest)",
    )
    parser.add_argument(
        "--format", nargs="+", choices=["pdf", "png"], default=["pdf", "png"],
        metavar="FORMAT",
        help="Output format(s): pdf, png, or both (default: pdf png)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        latest = runs_root / "latest"
        if not latest.exists():
            parser.error("No run_dir given and runs/latest does not exist. Run evaluate.py first.")
        run_dir = latest.resolve()

    if not run_dir.is_dir():
        parser.error(f"Not a directory: {run_dir}")

    export_report(run_dir, args.format)


if __name__ == "__main__":
    main()
