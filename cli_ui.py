#!/usr/bin/env python3
"""Terminal UI helpers: colors, progress bar, summary table, structured log output."""

import logging
import os
import sys

# ── Color support ──────────────────────────────────────────────────────────────

_USE_COLOR = (
    sys.stderr.isatty()
    and not os.environ.get("NO_COLOR")
    and os.environ.get("TERM", "") != "dumb"
)


def _e(code: str) -> str:
    return code if _USE_COLOR else ""


RESET         = _e("\033[0m")
BOLD          = _e("\033[1m")
DIM           = _e("\033[2m")
GREY          = _e("\033[37m")    # standard white — readable light grey on dark terminals
CYAN          = _e("\033[36m")
BLUE          = _e("\033[34m")
BRIGHT_RED    = _e("\033[91m")
BRIGHT_GREEN  = _e("\033[92m")
BRIGHT_YELLOW = _e("\033[93m")
BRIGHT_BLUE   = _e("\033[94m")
BRIGHT_CYAN   = _e("\033[96m")


def acc_color(val: float) -> str:
    """Return ANSI color for an accuracy or F1 value."""
    if val >= 0.90:
        return BRIGHT_GREEN
    if val >= 0.75:
        return BRIGHT_YELLOW
    return BRIGHT_RED


def render_bar(value: float, width: int = 20) -> str:
    """Unicode block progress bar, color-coded when terminal supports it."""
    filled = max(0, min(width, int(value * width)))
    empty = width - filled
    if _USE_COLOR:
        return f"\033[32m{'█' * filled}\033[90m{'░' * empty}{RESET}"
    return f"{'█' * filled}{'░' * empty}"


# ── Logging ────────────────────────────────────────────────────────────────────

_progress_active = False


class _CLIHandler(logging.StreamHandler):
    """Stderr log handler that clears the in-place progress bar before printing."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.levelno >= logging.ERROR:
                icon = f"{BRIGHT_RED}✗{RESET}"
                text = f"  {icon}  {msg}"
            elif record.levelno >= logging.WARNING:
                icon = f"{BRIGHT_YELLOW}⚠{RESET}"
                text = f"  {icon}  {msg}"
            else:
                icon = f"{GREY}·{RESET}"
                text = f"  {icon}  {DIM}{msg}{RESET}"

            if _USE_COLOR and _progress_active:
                self.stream.write("\r\033[K")  # clear in-place progress line

            self.stream.write(text + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    """Replace the default logging configuration with clean, icon-prefixed CLI output."""
    logging.root.handlers = []
    logging.root.addHandler(_CLIHandler(sys.stderr))
    logging.root.setLevel(logging.INFO)


# ── Startup banner ─────────────────────────────────────────────────────────────


def print_banner(
    run_tag: str,
    input_dir: object,
    run_dir: object,
    api_label: str,
    n_samples: int,
    workers: int,
) -> None:
    W = 62
    sep = f"{BRIGHT_BLUE}{'─' * W}{RESET}"
    print(f"\n{sep}", file=sys.stderr)
    print(f"{BOLD}{BRIGHT_BLUE}  ALD Model Evaluation Pipeline{RESET}", file=sys.stderr)
    print(sep, file=sys.stderr)
    for label, value in [
        ("Run",     run_tag),
        ("Input",   str(input_dir)),
        ("Output",  str(run_dir)),
        ("API",     api_label),
        ("Samples", f"{n_samples:,}"),
        ("Workers", str(workers)),
    ]:
        print(f"  {DIM}{label:<10}{RESET}{value}", file=sys.stderr)
    print(f"{sep}\n", file=sys.stderr)


# ── In-place progress line ─────────────────────────────────────────────────────


def print_progress(
    current: int,
    total: int,
    acc: float,
    lat_ms: float,
    errors: int,
    eta_str: str,
) -> None:
    global _progress_active
    _progress_active = True
    pct = current / total
    bar = render_bar(pct, width=24)
    acc_str = f"{acc_color(acc)}{acc:.1%}{RESET}"
    err_str = f"{BRIGHT_RED}{errors}{RESET}" if errors else f"{GREY}0{RESET}"
    line = (
        f"  {bar}  {BOLD}{pct:>4.0%}{RESET}"
        f"  {GREY}{current:,}/{total:,}{RESET}"
        f"  acc={acc_str}"
        f"  lat={lat_ms:.0f}ms"
        f"  err={err_str}"
        f"  ETA={GREY}{eta_str}{RESET}"
    )
    if _USE_COLOR:
        print(f"\r{line}", end="", flush=True, file=sys.stderr)
    else:
        print(line, file=sys.stderr)
    if current >= total:
        _progress_active = False
        if _USE_COLOR:
            print(file=sys.stderr)


# ── Summary box ────────────────────────────────────────────────────────────────


def print_summary(metrics: dict) -> None:
    W = 64
    outer = f"{BOLD}{'━' * W}{RESET}"
    inner = f"  {GREY}{'─' * (W - 4)}{RESET}"

    overall_acc = metrics.get("overall_accuracy", 0.0)
    col = acc_color(overall_acc)

    print()
    print(outer)
    print(f"{BOLD}  ALD EVALUATION RESULTS{RESET}")
    print(outer)

    valid   = metrics.get("valid_files", 0)
    total_f = metrics.get("total_files", 0)
    errors  = metrics.get("error_files", 0)

    if errors:
        err_tag = f"  {BRIGHT_RED}✗ {errors} error{'s' if errors != 1 else ''}{RESET}"
    else:
        err_tag = f"  {BRIGHT_GREEN}✓ no errors{RESET}"

    print(f"  Files       {BOLD}{valid:,}{RESET} / {total_f:,} valid{err_tag}")
    print(f"  Accuracy    {col}{BOLD}{overall_acc:.1%}{RESET}  {render_bar(overall_acc, 22)}")
    print(
        f"  Latency     avg {metrics.get('avg_latency_ms', 0):.0f} ms"
        f" · p50 {metrics.get('p50_latency_ms', 0):.0f} ms"
        f" · p95 {metrics.get('p95_latency_ms', 0):.0f} ms"
    )
    print(f"  Confidence  {metrics.get('avg_confidence', 0):.3f}")

    per_lang = metrics.get("per_language", {})
    if per_lang:
        print()
        print(f"  {BOLD}Per-Language{RESET}  {GREY}F1 · Precision · Recall · Support{RESET}")
        print(inner)
        for lang, m in sorted(per_lang.items()):
            f1 = m["f1"]
            c = acc_color(f1)
            print(
                f"    {BOLD}{lang}{RESET}  {render_bar(f1, 18)}  "
                f"{c}F1={f1:.3f}{RESET}"
                f"  prec={m['precision']:.3f}"
                f"  rec={m['recall']:.3f}"
                f"  {GREY}n={m['support']}{RESET}"
            )

    per_dur = metrics.get("per_duration", {})
    if per_dur:
        print()
        print(f"  {BOLD}Per-Duration{RESET}  {GREY}Accuracy · Count · Avg Confidence{RESET}")
        print(inner)
        for dur, m in sorted(per_dur.items()):
            acc = m["accuracy"]
            c = acc_color(acc)
            print(
                f"    {BOLD}{dur:>3}{RESET}  {render_bar(acc, 18)}  "
                f"{c}{acc:.1%}{RESET}"
                f"  {GREY}n={m['count']}  conf={m['avg_confidence']:.3f}{RESET}"
            )

    print()
    print(outer)
    print()


# ── Post-run status messages ───────────────────────────────────────────────────


def print_run_saved(run_dir: object, runs_dir: object) -> None:
    print(f"  {BRIGHT_GREEN}✓{RESET}  Saved   →  {BOLD}{run_dir}{RESET}", file=sys.stderr)
    print(f"  {BRIGHT_GREEN}✓{RESET}  Latest  →  {BOLD}{runs_dir}/latest{RESET}", file=sys.stderr)


def print_outputs_saved(output_dir: object, n_rows: int, accuracy: float) -> None:
    print(f"  {BRIGHT_GREEN}✓{RESET}  Output  →  {BOLD}{output_dir}/{RESET}", file=sys.stderr)
    print(f"     {GREY}predictions.csv   ({n_rows:,} rows){RESET}", file=sys.stderr)
    print(f"     {GREY}metrics.json      (accuracy={accuracy:.1%}){RESET}", file=sys.stderr)
    print(f"     {GREY}report_data.json  (HTML source){RESET}", file=sys.stderr)


def print_view_report(project_dir: object) -> None:
    print()
    print(f"  {BOLD}View report:{RESET}")
    print(f"    cd {CYAN}{project_dir}{RESET}")
    print(f"    {CYAN}python3 -m http.server 8080{RESET}")
    print(f"    open {CYAN}http://localhost:8080/{RESET}")
    print()
    print(f"  {GREY}Or re-run with {RESET}{BOLD}--serve{RESET}{GREY} to open automatically{RESET}")
    print()


def print_serve_ready(url: str) -> None:
    print()
    print(f"  {BRIGHT_GREEN}●{RESET}  {BOLD}Report server ready{RESET}")
    print(f"     {CYAN}{url}{RESET}")
    print(f"  {GREY}  Ctrl+C to stop{RESET}")
    print()
