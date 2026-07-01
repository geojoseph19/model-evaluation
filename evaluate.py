#!/usr/bin/env python3
"""ALD (Audio Language Detection) Model Evaluation Pipeline.

Usage:
    python3 evaluate.py                                              # dummy/simulation
    python3 evaluate.py --api-url http://host/detect                 # real API
    python3 evaluate.py --api-url http://host/detect --tag v1.2      # named run
    python3 evaluate.py --api-url http://host/detect -j 5 --serve    # throttle + open report

Runs are saved under runs/<timestamp>_<tag>/ and runs/latest always
points to the most recent run.
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["bn", "gu", "kn", "ml", "mr", "ta", "te"]
DURATION_MAP = {1: "1s", 2: "2s", 3: "3s", 5: "5s"}

# ---------------------------------------------------------------------------
# Configuration — edit these defaults or override via CLI flags
# ---------------------------------------------------------------------------

# Set your ALD API endpoint here. --api-url overrides this at runtime.
# Leave empty ("") to default to dummy/simulation mode.
ALD_API_URL = ""

# ---------------------------------------------------------------------------
# Auto-tag generator
# ---------------------------------------------------------------------------

def _auto_tag(runs_dir: Path) -> str:
    """Return next incremental version tag (v1, v2, …) based on existing runs."""
    highest = 0
    if runs_dir.exists():
        import re
        for entry in runs_dir.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                continue
            m = re.search(r"_v(\d+)$", entry.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"v{highest + 1}"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    input_dir: Path
    output_dir: Path        # resolved run directory (runs/<ts>_<tag>/)
    api_url: str = "http://localhost:8000/detect"
    use_dummy: bool = True
    concurrency: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    tag: str = ""           # optional human label, e.g. "v1.2" or "baseline"


@dataclass
class EvalSample:
    file_id: str          # stem: bn_00001_2s
    filename: str         # bn_00001_2s.wav
    path: Path
    ground_truth: str     # bn
    duration: str         # 2s
    language: str         # bn (folder name)
    sample_rate: Optional[int] = None
    codec: Optional[str] = None


@dataclass
class EvalResult:
    file_id: str
    filename: str
    ground_truth: str
    predicted: str
    confidence: float
    duration: str
    language: str
    latency_ms: float
    correct: bool
    error: Optional[str] = None
    all_scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ALD API client (dummy + real)
# ---------------------------------------------------------------------------

class ALDClient:
    def __init__(self, config: EvalConfig):
        self.config = config
        self._session = None

    async def __aenter__(self):
        if not self.config.use_dummy:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise RuntimeError("aiohttp required for real API: pip install aiohttp")
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def detect(self, sample: EvalSample) -> dict:
        if self.config.use_dummy:
            return await self._dummy_detect(sample)
        return await self._real_detect(sample)

    async def _dummy_detect(self, sample: EvalSample) -> dict:
        """Simulate ALD API with duration-correlated accuracy."""
        await asyncio.sleep(random.uniform(0.04, 0.25))

        # Accuracy improves with clip length (realistic ALD behaviour)
        accuracy_by_duration = {"1s": 0.72, "2s": 0.81, "3s": 0.88, "5s": 0.93}
        accuracy = accuracy_by_duration.get(sample.duration, 0.85)

        if random.random() < accuracy:
            detected = sample.ground_truth
            confidence = random.uniform(0.75, 0.97)
        else:
            others = [l for l in SUPPORTED_LANGUAGES if l != sample.ground_truth]
            detected = random.choice(others)
            confidence = random.uniform(0.35, 0.70)

        # Construct plausible score distribution
        scores = {l: round(random.uniform(0.01, 0.12), 4) for l in SUPPORTED_LANGUAGES}
        scores[detected] = round(confidence, 4)
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}

        return {
            "detected_language": detected,
            "confidence": round(confidence, 4),
            "all_scores": scores,
        }

    async def _real_detect(self, sample: EvalSample) -> dict:
        """POST audio/wav to ALD API, return JSON response.

        Expected response schema:
            {
                "detected_language": "bn",
                "confidence": 0.92,
                "all_scores": {"bn": 0.92, "ml": 0.03, ...}   // optional
            }
        """
        import aiohttp

        for attempt in range(self.config.retry_attempts):
            try:
                audio_bytes = sample.path.read_bytes()
                form = aiohttp.FormData()
                form.add_field(
                    "audio",
                    audio_bytes,
                    filename=sample.filename,
                    content_type="audio/wav",
                )
                async with self._session.post(
                    self.config.api_url,
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if attempt == self.config.retry_attempts - 1:
                    raise
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt+1} for {sample.filename}: {exc} (wait {wait:.1f}s)")
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def load_metadata(input_dir: Path) -> dict[str, dict]:
    """Return dict keyed by filename → metadata row."""
    meta_path = input_dir / "metadata.csv"
    if not meta_path.exists():
        logger.warning("metadata.csv not found — ground truth inferred from folder names")
        return {}

    records = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("file", "").strip()
            if fname:
                records[fname] = row
    logger.info(f"Loaded {len(records)} records from metadata.csv")
    return records


def discover_samples(config: EvalConfig) -> list[EvalSample]:
    metadata = load_metadata(config.input_dir)
    samples: list[EvalSample] = []

    for lang_dir in sorted(config.input_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name not in SUPPORTED_LANGUAGES:
            continue
        for dur_dir in sorted(lang_dir.iterdir()):
            if not dur_dir.is_dir():
                continue
            for wav in sorted(dur_dir.glob("*.wav")):
                row = metadata.get(wav.name, {})

                # Ground truth: metadata > folder name fallback
                ground_truth = row.get("language", lang_dir.name).strip().lower()

                # Duration: metadata > parse folder name
                dur_sec = row.get("duration_sec", "").strip()
                if dur_sec and dur_sec.isdigit():
                    duration = DURATION_MAP.get(int(dur_sec), dur_dir.name)
                else:
                    duration = dur_dir.name  # already "2s", "3s", etc.

                samples.append(EvalSample(
                    file_id=wav.stem,
                    filename=wav.name,
                    path=wav,
                    ground_truth=ground_truth,
                    duration=duration,
                    language=lang_dir.name,
                    sample_rate=int(row["sample_rate"]) if row.get("sample_rate", "").isdigit() else None,
                    codec=row.get("codec"),
                ))

    logger.info(f"Discovered {len(samples)} audio files")
    return samples


# ---------------------------------------------------------------------------
# Async evaluation engine
# ---------------------------------------------------------------------------

async def _process_one(
    sample: EvalSample,
    client: ALDClient,
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.detect(sample)
            latency_ms = (time.perf_counter() - t0) * 1000

            predicted = str(resp.get("detected_language", "unknown")).lower()
            confidence = float(resp.get("confidence", 0.0))
            all_scores = resp.get("all_scores", {})

            return EvalResult(
                file_id=sample.file_id,
                filename=sample.filename,
                ground_truth=sample.ground_truth,
                predicted=predicted,
                confidence=confidence,
                duration=sample.duration,
                language=sample.language,
                latency_ms=round(latency_ms, 2),
                correct=(predicted == sample.ground_truth),
                all_scores=all_scores,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"  FAILED {sample.filename}: {exc}")
            return EvalResult(
                file_id=sample.file_id,
                filename=sample.filename,
                ground_truth=sample.ground_truth,
                predicted="error",
                confidence=0.0,
                duration=sample.duration,
                language=sample.language,
                latency_ms=round(latency_ms, 2),
                correct=False,
                error=str(exc),
            )


async def run_evaluation(samples: list[EvalSample], config: EvalConfig) -> list[EvalResult]:
    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[EvalResult] = []
    total = len(samples)
    log_interval = max(1, total // 10)

    async with ALDClient(config) as client:
        tasks = [_process_one(s, client, semaphore) for s in samples]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)
            if i % log_interval == 0 or i == total:
                errors = sum(1 for r in results if r.error)
                logger.info(f"  {i}/{total} done  |  errors: {errors}")

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[EvalResult]) -> dict:
    valid = [r for r in results if not r.error]
    if not valid:
        return {"error": "No valid results to compute metrics"}

    y_true = [r.ground_truth for r in valid]
    y_pred = [r.predicted for r in valid]
    labels = sorted(set(y_true) | (set(y_pred) - {"error", "unknown"}))
    durations = sorted({r.duration for r in valid})

    # --- per-language metrics ---
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )

        overall_accuracy = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_language = {
            lang: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i, lang in enumerate(labels)
        }
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    except ImportError:
        # stdlib fallback
        overall_accuracy = sum(r.correct for r in valid) / len(valid)
        per_language = {}
        cm = []
        for lang in labels:
            tp = sum(1 for r in valid if r.ground_truth == lang and r.predicted == lang)
            fp = sum(1 for r in valid if r.ground_truth != lang and r.predicted == lang)
            fn = sum(1 for r in valid if r.ground_truth == lang and r.predicted != lang)
            support = sum(1 for r in valid if r.ground_truth == lang)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r_ = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
            per_language[lang] = {
                "precision": round(p, 4),
                "recall": round(r_, 4),
                "f1": round(f, 4),
                "support": support,
            }
        cm = [
            [sum(1 for r in valid if r.ground_truth == tl and r.predicted == pl) for pl in labels]
            for tl in labels
        ]

    # --- per-duration ---
    per_duration = {}
    for dur in durations:
        subset = [r for r in valid if r.duration == dur]
        if not subset:
            continue
        per_duration[dur] = {
            "accuracy": round(sum(r.correct for r in subset) / len(subset), 4),
            "count": len(subset),
            "avg_confidence": round(sum(r.confidence for r in subset) / len(subset), 4),
        }

    # --- cross breakdown: language × duration accuracy ---
    cross_breakdown = {}
    for lang in labels:
        cross_breakdown[lang] = {}
        for dur in durations:
            subset = [r for r in valid if r.language == lang and r.duration == dur]
            if subset:
                cross_breakdown[lang][dur] = round(
                    sum(r.correct for r in subset) / len(subset), 4
                )

    # --- latency / confidence stats ---
    latencies = sorted(r.latency_ms for r in valid)
    confidences = [r.confidence for r in valid]
    n = len(latencies)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(results),
        "valid_files": len(valid),
        "error_files": len(results) - len(valid),
        "overall_accuracy": round(overall_accuracy, 4),
        "per_language": per_language,
        "per_duration": per_duration,
        "cross_breakdown": cross_breakdown,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "p50_latency_ms": round(latencies[n // 2], 2),
        "p95_latency_ms": round(latencies[int(0.95 * n)], 2),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "labels": labels,
        "durations": durations,
    }


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def make_run_dir(runs_dir: Path, tag: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_info(run_dir: Path, config: EvalConfig, elapsed: float, n_files: int) -> None:
    info: dict = {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": config.tag,
        "input_dir": str(config.input_dir.resolve()),
        "api": "dummy" if config.use_dummy else config.api_url,
        "workers": config.concurrency,
        "total_files": n_files,
        "elapsed_seconds": round(elapsed, 2),
    }
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_hash"] = git_hash
    except Exception:
        pass
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    latest = runs_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        # Windows fallback — write a pointer file instead
        (runs_dir / "latest.txt").write_text(run_dir.name, encoding="utf-8")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_results(results: list[EvalResult], metrics: dict, output_dir: Path) -> None:

    # predictions.csv — one row per file
    csv_path = output_dir / "predictions.csv"
    csv_fields = [
        "file_id", "filename", "ground_truth", "predicted",
        "confidence", "duration", "language", "latency_ms", "correct", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.language, x.duration, x.file_id)):
            writer.writerow({k: getattr(r, k, "") for k in csv_fields})

    # metrics.json — aggregated stats
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # report_data.json — consumed by index.html
    report_data = {
        "metrics": metrics,
        "predictions": [
            {k: getattr(r, k) for k in csv_fields}
            for r in sorted(results, key=lambda x: (x.language, x.duration, x.file_id))
        ],
    }
    with open(output_dir / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f)

    logger.info(f"Saved results → {output_dir}/")
    logger.info(f"  predictions.csv ({len(results)} rows)")
    logger.info(f"  metrics.json (accuracy={metrics.get('overall_accuracy', 'N/A')})")
    logger.info(f"  report_data.json (HTML report source)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(metrics: dict) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print("  ALD EVALUATION SUMMARY")
    print(sep)
    print(f"  Files:       {metrics['valid_files']}/{metrics['total_files']} valid")
    print(f"  Accuracy:    {metrics['overall_accuracy']:.1%}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.0f} ms  (p95: {metrics['p95_latency_ms']:.0f} ms)")
    print(f"  Avg conf:    {metrics['avg_confidence']:.3f}")
    print(f"\n  Per-language F1 / support:")
    for lang, m in sorted(metrics.get("per_language", {}).items()):
        bar = "█" * int(m["f1"] * 20)
        print(f"    {lang}  {bar:<20}  F1={m['f1']:.3f}  n={m['support']}")
    print(f"\n  Per-duration accuracy:")
    for dur, m in sorted(metrics.get("per_duration", {}).items()):
        bar = "█" * int(m["accuracy"] * 20)
        print(f"    {dur}  {bar:<20}  {m['accuracy']:.1%}  (n={m['count']})")
    print(sep)


def export_report(run_dir: Path, formats: list[str]) -> None:
    """Render index.html in headless Chromium and export as PDF and/or PNG."""
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

    report_data = (run_dir / "report_data.json").read_text(encoding="utf-8")
    logger.info(f"Exporting report as: {', '.join(formats)}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Intercept the fetch for report_data.json — serve from run dir, no HTTP server needed
        def handle_route(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=report_data,
            )

        page.route("**/report_data.json", handle_route)
        page.goto(f"file://{report_html.resolve()}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)  # let Chart.js finish rendering

        if "pdf" in formats:
            pdf_path = run_dir / "report.pdf"
            page.pdf(
                path=str(pdf_path),
                format="A3",
                landscape=True,
                print_background=True,
            )
            logger.info(f"  PDF  → {pdf_path}")

        if "png" in formats:
            png_path = run_dir / "report.png"
            page.screenshot(path=str(png_path), full_page=True)
            logger.info(f"  PNG  → {png_path}")

        browser.close()


def serve_report(output_dir: Path, port: int = 8080) -> None:
    import http.server
    import threading
    import webbrowser

    serve_root = Path(__file__).parent.resolve()
    os.chdir(serve_root)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # silence access logs

    data_path = output_dir.resolve().relative_to(serve_root) / "report_data.json"
    url = f"http://localhost:{port}/?data={data_path}"

    with http.server.HTTPServer(("", port), handler) as httpd:
        print(f"\n  Report server running.")
        print(f"  Open: {url}")
        print(f"  Stop: Ctrl+C\n")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="eval_clips_telephony",
        help="Path to evaluation clips directory (default: eval_clips_telephony)",
    )
    parser.add_argument(
        "--api-url", default=None, metavar="URL",
        help="ALD API endpoint (e.g. http://host/detect). Omit to run in dummy/simulation mode.",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=10, dest="workers", metavar="N",
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--tag", default=None, metavar="TAG",
        help="Label for this run, e.g. 'v1.2' or 'baseline' (auto-generated if omitted)",
    )
    parser.add_argument(
        "--runs-dir", default=None, metavar="DIR",
        help="Root directory for all runs (default: ./runs next to this script)",
    )
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server and open report after evaluation")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port when using --serve (default: 8080)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        parser.error(f"Input directory not found: {input_dir}")

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(__file__).parent / "runs"

    # API URL: CLI flag > ALD_API_URL constant > dummy mode
    api_url = args.api_url or ALD_API_URL or None
    use_dummy = api_url is None

    # Tag: explicit > auto-incremented version
    tag = args.tag or _auto_tag(runs_dir)

    # Create the timestamped run directory before evaluation starts
    run_dir = make_run_dir(runs_dir, tag)

    config = EvalConfig(
        input_dir=input_dir,
        output_dir=run_dir,
        api_url=api_url or "http://localhost:8000/detect",
        use_dummy=use_dummy,
        concurrency=args.workers,
        tag=tag,
    )

    logger.info(f"ALD Evaluation Pipeline")
    logger.info(f"  Input dir:   {config.input_dir}")
    logger.info(f"  Run dir:     {run_dir}")
    logger.info(f"  API:         {'DUMMY (simulated)' if config.use_dummy else config.api_url}")
    logger.info(f"  Concurrency: {config.concurrency} parallel requests")

    samples = discover_samples(config)
    if not samples:
        logger.error("No .wav files found. Check directory structure.")
        return

    t0 = time.time()
    results = asyncio.run(run_evaluation(samples, config))
    elapsed = time.time() - t0
    throughput = len(results) / elapsed
    logger.info(f"Done: {len(results)} files in {elapsed:.1f}s ({throughput:.1f} files/s)")

    metrics = compute_metrics(results)
    save_results(results, metrics, run_dir)
    write_run_info(run_dir, config, elapsed, len(results))
    update_latest_symlink(runs_dir, run_dir)
    print_summary(metrics)

    logger.info(f"Run saved → {run_dir}")
    logger.info(f"Symlink   → {runs_dir}/latest")

    if args.serve:
        serve_report(run_dir, port=args.port)
    else:
        print(f"\n  View report:")
        print(f"    python3 -m http.server 8080  (from {Path(__file__).parent})")
        print(f"    open http://localhost:8080/")
        print(f"\n  Or: python3 evaluate.py --serve\n")


if __name__ == "__main__":
    main()
