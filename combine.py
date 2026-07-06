#!/usr/bin/env python3
"""Merge multiple ALD evaluation runs into one combined report.

Usage:
    python3 combine.py runs/2026-07-01_v1 runs/2026-07-01_v2
    python3 combine.py runs/2026-07-01_v1 runs/2026-07-01_v2 --tag combined-v1v2
    python3 combine.py runs/2026-07-01_v1 runs/2026-07-01_v2 --export pdf png
    python3 combine.py runs/2026-07-01_v1 runs/2026-07-01_v2 --serve
    python3 combine.py --all
    python3 combine.py --all --tag full-eval --export pdf png

Duplicate file_ids (same clip evaluated in multiple runs) keep the LAST run's
result by default.  Pass --keep first to keep the first occurrence instead.

Output is saved under runs/<timestamp>_<tag>/ just like a normal evaluate.py run.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cli_ui

logger = logging.getLogger(__name__)


def load_predictions(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "predictions.csv"
    if not csv_path.exists():
        logger.warning(f"No predictions.csv in {run_dir} — skipping")
        return []

    # all_scores is not in predictions.csv — load from report_data.json if available
    all_scores_map: dict[str, dict] = {}
    report_data_path = run_dir / "report_data.json"
    if report_data_path.exists():
        try:
            report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
            for pred in report_data.get("predictions", []):
                fid = pred.get("file_id")
                if fid:
                    all_scores_map[fid] = pred.get("all_scores", {})
        except Exception:
            pass

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["all_scores"] = all_scores_map.get(row["file_id"], {})
            rows.append(row)
    return rows


def row_to_eval_result(row: dict):
    """Convert a CSV row (augmented with all_scores) back to an EvalResult."""
    from evaluate import EvalResult
    return EvalResult(
        file_id=row["file_id"],
        filename=row["filename"],
        ground_truth=row["ground_truth"],
        predicted=row["predicted"],
        confidence=float(row["confidence"]) if row["confidence"] else 0.0,
        duration=row["duration"],
        language=row["language"],
        latency_ms=float(row["latency_ms"]) if row["latency_ms"] else 0.0,
        correct=row["correct"].strip().lower() == "true",
        error=row["error"] if row.get("error") else None,
        all_scores=row.get("all_scores", {}),
    )


def merge_runs(run_dirs: list[Path], keep_first: bool) -> list:
    seen: dict[str, object] = {}
    for run_dir in run_dirs:
        rows = load_predictions(run_dir)
        if not rows:
            continue
        logger.info(f"Loaded {len(rows)} predictions from {run_dir.name}")
        for row in rows:
            fid = row["file_id"]
            if keep_first and fid in seen:
                continue
            seen[fid] = row_to_eval_result(row)

    results = list(seen.values())
    logger.info(f"Merged total: {len(results)} unique predictions")
    return results


def main() -> None:
    runs_root = Path(__file__).parent / "runs"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="*", help="Run directories to merge (omit to use --all)")
    parser.add_argument("--all", action="store_true", help="Merge every run in the runs/ directory")
    parser.add_argument("--runs-dir", default=None, metavar="DIR",
                        help="Root runs directory for --all (default: ./runs)")
    parser.add_argument("--tag", default="combined", help="Tag suffix for the output run dir (default: combined)")
    parser.add_argument("--keep", choices=["first", "last"], default="last",
                        help="When a file_id appears in multiple runs, keep first or last (default: last)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip automatic PDF/PNG export")
    parser.add_argument("--export", nargs="+", choices=["pdf", "png"], metavar="FORMAT",
                        help="Export specific formats only (default: pdf png). Use --no-export to skip entirely.")
    parser.add_argument("--serve", action="store_true", help="Open the report in a browser after merging")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (used with --serve, default: 8080)")
    args = parser.parse_args()

    cli_ui.setup_logging()

    if args.runs_dir:
        runs_root = Path(args.runs_dir).resolve()

    resolved: list[Path] = []
    if args.all:
        if not runs_root.is_dir():
            parser.error(f"Runs directory not found: {runs_root}")
        resolved = sorted(
            d for d in runs_root.iterdir()
            if d.is_dir() and not d.is_symlink() and not (d / "combine_meta.json").exists()
        )
        if not resolved:
            parser.error(f"No run directories found in {runs_root}")
        if len(resolved) == 1:
            logger.warning("Only one run found — combined output will be identical to the source run.")
        logger.info(f"--all: found {len(resolved)} runs in {runs_root}")
    else:
        if not args.run_dirs:
            parser.error("Provide run directories or use --all to merge everything in runs/.")
        for p in args.run_dirs:
            d = Path(p).resolve()
            if not d.is_dir():
                parser.error(f"Not a directory: {p}")
            resolved.append(d)
        if len(resolved) < 2:
            parser.error("Provide at least two run directories to merge.")

    from evaluate import compute_metrics, save_results, print_summary, update_latest_symlink, serve_report

    results = merge_runs(resolved, keep_first=(args.keep == "first"))
    if not results:
        logger.error("No predictions loaded — nothing to combine.")
        sys.exit(1)

    metrics = compute_metrics(results)
    if "error" in metrics:
        logger.error(f"Cannot compute metrics: {metrics['error']}")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    tag = args.tag.strip() or "combined"
    out_dir = runs_root / f"{ts}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write combine_meta.json immediately so --all skips this dir even if a later step crashes
    meta = {
        "combined_from": [str(d) for d in resolved],
        "keep_policy": args.keep,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "combine_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_results(results, metrics, out_dir)
    print_summary(metrics)

    update_latest_symlink(runs_root, out_dir)
    logger.info(f"Combined run saved → {out_dir}")

    if not args.no_export:
        from export import export_report
        export_report(out_dir, args.export or ["pdf", "png"])

    if args.serve:
        serve_report(out_dir, port=args.port)


if __name__ == "__main__":
    main()
