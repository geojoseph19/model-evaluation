#!/usr/bin/env python3
"""ALD (Audio Language Detection) Model Evaluation Pipeline.

Usage:
    python3 evaluate.py                                              # dummy/simulation
    python3 evaluate.py --api-url http://host/detect                 # real API
    python3 evaluate.py --api-url http://host/detect --tag v1.2      # named run
    python3 evaluate.py --api-url http://host/detect -j 5 --serve    # throttle + open report
    python3 evaluate.py --no-export                                  # skip PDF/PNG export

Runs are saved under runs/<timestamp>_<tag>/ and runs/latest always
points to the most recent run.
"""

import asyncio
import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import cli_ui
from ald_client import ALDClient, LocalModelClient
from export import export_report
from system_info import (
    collect_device_info,
    collect_runtime_info,
    take_pre_run_snapshot,
    collect_peak_usage,
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

def _parse_int(value: str) -> Optional[int]:
    try:
        return int(float(value.strip())) if value.strip() else None
    except (ValueError, TypeError):
        return None


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
    local_model_dir: Optional[str] = None  # path to LID-version-2.0 dir; enables LocalModelClient


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
                if dur_sec:
                    try:
                        duration = DURATION_MAP.get(int(float(dur_sec)), dur_dir.name)
                    except ValueError:
                        duration = dur_dir.name
                else:
                    duration = dur_dir.name  # already "2s", "3s", etc.

                if ground_truth not in SUPPORTED_LANGUAGES:
                    logger.warning(
                        f"Unexpected ground truth '{ground_truth}' for {wav.name} — not in SUPPORTED_LANGUAGES"
                    )

                samples.append(EvalSample(
                    file_id=wav.stem,
                    filename=wav.name,
                    path=wav,
                    ground_truth=ground_truth,
                    duration=duration,
                    language=lang_dir.name,
                    sample_rate=_parse_int(row.get("sample_rate", "")),
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
            resp = await client.detect(sample.path, sample.filename)
            latency_ms = (time.perf_counter() - t0) * 1000

            return EvalResult(
                file_id=sample.file_id,
                filename=sample.filename,
                ground_truth=sample.ground_truth,
                predicted=resp["detected_language"],
                confidence=resp["confidence"],
                duration=sample.duration,
                language=sample.language,
                latency_ms=round(latency_ms, 2),
                correct=(resp["detected_language"] == sample.ground_truth),
                all_scores=resp["all_scores"],
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
    log_interval = max(1, total // 100)   # log every ~1% instead of ~10%
    start_time = time.monotonic()

    # running per-language correct/total counters
    lang_correct: dict[str, int] = {}
    lang_total: dict[str, int] = {}

    if config.local_model_dir:
        client_ctx = LocalModelClient(config.local_model_dir)
    else:
        client_ctx = ALDClient(
            api_url=config.api_url,
            use_dummy=config.use_dummy,
            retry_attempts=config.retry_attempts,
            retry_delay=config.retry_delay,
        )
    async with client_ctx as client:
        tasks = [_process_one(s, client, semaphore) for s in samples]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)

            lang = result.ground_truth
            if lang not in ("error", "unknown"):
                lang_total[lang] = lang_total.get(lang, 0) + 1
                if result.correct:
                    lang_correct[lang] = lang_correct.get(lang, 0) + 1

            if i % log_interval == 0 or i == total:
                errors = sum(1 for r in results if r.error)
                elapsed = time.monotonic() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_s = (total - i) / rate if rate > 0 else 0
                if eta_s >= 86400:
                    eta_str = f"{int(eta_s // 86400)}d{int(eta_s % 86400 // 3600):02d}h"
                elif eta_s >= 3600:
                    eta_str = f"{int(eta_s // 3600)}h{int(eta_s % 3600 // 60):02d}m"
                else:
                    eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                valid = [r for r in results if not r.error and r.ground_truth not in ("error", "unknown")]
                running_acc = sum(1 for r in valid if r.correct) / len(valid) if valid else 0.0
                avg_lat = sum(r.latency_ms for r in valid) / len(valid) if valid else 0.0
                cli_ui.print_progress(i, total, running_acc, avg_lat, errors, eta_str)

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[EvalResult]) -> dict:
    valid = [r for r in results if not r.error]
    if not valid:
        return {"error": "No valid results to compute metrics"}

    _excluded = {"error", "unknown"}
    eval_set = [r for r in valid if r.ground_truth not in _excluded]
    y_true = [r.ground_truth for r in eval_set]
    y_pred = [r.predicted for r in eval_set]
    labels = sorted((set(y_true) - _excluded) | (set(y_pred) - _excluded))
    durations = sorted({r.duration for r in eval_set})

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
        overall_accuracy = sum(r.correct for r in eval_set) / len(eval_set) if eval_set else 0.0
        per_language = {}
        cm = []
        for lang in labels:
            tp = sum(1 for r in eval_set if r.ground_truth == lang and r.predicted == lang)
            fp = sum(1 for r in eval_set if r.ground_truth != lang and r.predicted == lang)
            fn = sum(1 for r in eval_set if r.ground_truth == lang and r.predicted != lang)
            support = sum(1 for r in eval_set if r.ground_truth == lang)
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
            [sum(1 for r in eval_set if r.ground_truth == tl and r.predicted == pl) for pl in labels]
            for tl in labels
        ]

    # --- per-duration ---
    per_duration = {}
    for dur in durations:
        subset = [r for r in eval_set if r.duration == dur]
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
            subset = [r for r in eval_set if r.language == lang and r.duration == dur]
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
        "p50_latency_ms": round(
            (latencies[n // 2 - 1] + latencies[n // 2]) / 2 if n % 2 == 0 else latencies[n // 2], 2
        ),
        "p95_latency_ms": round(latencies[min(math.ceil(0.95 * n) - 1, n - 1)], 2),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "labels": labels,
        "durations": durations,
    }


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def make_run_dir(runs_dir: Path, tag: str) -> Path:
    import socket
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    host = socket.gethostname().split(".")[0]  # short hostname, no FQDN
    name = f"{ts}_{host}_{tag}" if tag else f"{ts}_{host}"
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_info(
    run_dir: Path,
    config: EvalConfig,
    elapsed: float,
    n_files: int,
    pre_run: Optional[dict] = None,
    peak_usage: Optional[dict] = None,
) -> None:
    import socket
    import subprocess

    info: dict = {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": config.tag,
        "hostname": socket.gethostname(),
        "input_dir": str(config.input_dir.resolve()),
        "api": f"local:{config.local_model_dir}" if config.local_model_dir else ("dummy" if config.use_dummy else config.api_url),
        "workers": config.concurrency,
        "retry_attempts": config.retry_attempts,
        "retry_delay": config.retry_delay,
        "total_files": n_files,
        "elapsed_seconds": round(elapsed, 2),
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        info["git_hash"] = git_hash
    except Exception:
        pass

    info["runtime"] = collect_runtime_info()

    device = collect_device_info(config.input_dir)
    if pre_run:
        device["at_run_start"] = pre_run
    if peak_usage:
        device["peak_usage"] = peak_usage
    if device:
        info["device"] = device

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
            {**{k: getattr(r, k) for k in csv_fields}, "all_scores": r.all_scores}
            for r in sorted(results, key=lambda x: (x.language, x.duration, x.file_id))
        ],
    }
    with open(output_dir / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f)

    cli_ui.print_outputs_saved(output_dir, len(results), metrics.get("overall_accuracy", 0.0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(metrics: dict) -> None:
    cli_ui.print_summary(metrics)


def serve_report(output_dir: Path, port: int = 8080) -> None:
    import http.server
    import threading
    import webbrowser

    serve_root = Path(__file__).parent.resolve()

    try:
        data_path = output_dir.resolve().relative_to(serve_root) / "report_data.json"
    except ValueError:
        # output_dir is outside the project root; symlink report_data.json into a temp
        # location under serve_root so SimpleHTTPRequestHandler can serve it
        import tempfile, shutil, atexit
        tmp_dir = Path(tempfile.mkdtemp(dir=serve_root, prefix=".report_"))
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        shutil.copy2(output_dir / "report_data.json", tmp_dir / "report_data.json")
        data_path = tmp_dir.relative_to(serve_root) / "report_data.json"

    url = f"http://localhost:{port}/?data={quote(str(data_path))}"

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_root), **kwargs)

        def log_message(self, *args):  # silence access logs
            pass

    with http.server.HTTPServer(("", port), _Handler) as httpd:
        cli_ui.print_serve_ready(url)
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
    parser.add_argument("--no-export", action="store_true",
                        help="Skip automatic PDF/PNG export after evaluation")
    parser.add_argument(
        "--local-model", default=None, metavar="DIR",
        help="Path to LID-version-2.0 directory; runs uvector WSSL model in-process instead of API",
    )
    args = parser.parse_args()

    cli_ui.setup_logging()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        parser.error(f"Input directory not found: {input_dir}")

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(__file__).parent / "runs"

    # API URL: CLI flag > ALD_API_URL constant > dummy mode
    api_url = args.api_url or ALD_API_URL or None
    use_dummy = api_url is None and not args.local_model

    # Tag: explicit > auto-incremented version
    tag = args.tag or _auto_tag(runs_dir)

    # Discover samples before creating the run dir so a bad input doesn't leave an orphan dir
    samples = discover_samples(EvalConfig(
        input_dir=input_dir,
        output_dir=runs_dir,  # placeholder; only input_dir is read during discovery
        api_url=api_url or "http://localhost:8000/detect",
        use_dummy=use_dummy,
        concurrency=args.workers,
        tag=tag,
    ))
    if not samples:
        logger.error(
            f"No .wav files found in: {input_dir}\n"
            f"         Expected:  {input_dir}/<lang>/<duration>/*.wav\n"
            f"         Example:   {input_dir}/bn/2s/bn_00001_2s.wav"
        )
        return

    run_dir = make_run_dir(runs_dir, tag)

    config = EvalConfig(
        input_dir=input_dir,
        output_dir=run_dir,
        api_url=api_url or "http://localhost:8000/detect",
        use_dummy=use_dummy,
        concurrency=args.workers,
        tag=tag,
        local_model_dir=args.local_model,
    )

    if config.local_model_dir:
        api_label = f"LOCAL ({config.local_model_dir})"
    elif config.use_dummy:
        api_label = "DUMMY (simulated)"
    else:
        api_label = config.api_url
    cli_ui.print_banner(config.tag, config.input_dir, run_dir, api_label, len(samples), config.concurrency)

    pre_run = take_pre_run_snapshot()
    t0 = time.time()
    results = asyncio.run(run_evaluation(samples, config))
    elapsed = time.time() - t0
    peak_usage = collect_peak_usage()
    throughput = len(results) / elapsed
    logger.info(f"Done: {len(results):,} files in {elapsed:.1f}s ({throughput:.1f} files/s)")

    metrics = compute_metrics(results)
    save_results(results, metrics, run_dir)
    write_run_info(run_dir, config, elapsed, len(results), pre_run=pre_run, peak_usage=peak_usage)
    update_latest_symlink(runs_dir, run_dir)
    if "error" in metrics:
        logger.error(f"Cannot compute metrics: {metrics['error']}")
    else:
        print_summary(metrics)

    cli_ui.print_run_saved(run_dir, runs_dir)

    if not args.no_export:
        export_report(run_dir, ["pdf", "png"])

    if args.serve:
        serve_report(run_dir, port=args.port)
    else:
        cli_ui.print_view_report(Path(__file__).parent)


if __name__ == "__main__":
    main()
