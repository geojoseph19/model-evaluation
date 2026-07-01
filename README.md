# ALD Model Evaluation

Evaluates an Audio Language Detection (ALD) model against labelled telephony clips.

## Setup

```bash
pip install scikit-learn aiohttp
```

## Directory structure

```
evaluation/
├── evaluate.py
├── report.html
├── eval_clips_telephony/          # input data — never modified by the script
│   ├── metadata.csv               # ground truth (file, language, duration_sec, …)
│   └── bn/  gu/  kn/  ml/  mr/  ta/  te/
│       └── 2s/  3s/  5s/
│           └── *.wav
└── runs/                          # all evaluation runs (created on first run)
    ├── 2026-07-01_110755_baseline/
    │   ├── run_info.json          # config snapshot: tag, api, workers, git hash
    │   ├── predictions.csv        # per-file: ground truth, predicted, confidence, latency
    │   ├── metrics.json           # overall accuracy, per-language P/R/F1, confusion matrix
    │   └── report_data.json       # consumed by report.html
    ├── 2026-07-02_093020_v1.2/
    │   └── …
    └── latest -> 2026-07-02_093020_v1.2   # symlink, always points to most recent run
```

## Run evaluation

```bash
# Dummy/simulation mode (no API needed)
python3 evaluate.py

# Real ALD API — tag the run for easy identification
python3 evaluate.py --api-url http://your-host/detect --tag v1.2

# Limit concurrency (default: 10)
python3 evaluate.py --api-url http://your-host/detect --tag v1.2 -j 5

# Evaluate and immediately open the report in a browser
python3 evaluate.py --api-url http://your-host/detect --tag v1.2 --serve
```

Each run creates a timestamped directory under `runs/` and updates the `runs/latest` symlink.

## Export report as PDF / PNG

Requires Playwright (one-time setup):
```bash
pip install playwright
playwright install chromium
```

```bash
# Export latest run (PDF + PNG)
python3 export.py

# Export a specific run
python3 export.py runs/2026-07-01_113219_baseline

# PDF only / PNG only
python3 export.py --format pdf
python3 export.py --format png
```

Exports are saved as `report.pdf` and `report.png` inside the run directory alongside the CSV/JSON files.

## View the report

```bash
# From the evaluation/ directory
python3 -m http.server 8080
```

Open **http://localhost:8080/report.html** in a browser.

Or run evaluation + serve in one step:

```bash
python3 evaluate.py eval_clips_telephony/ --serve
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `input_dir` | `eval_clips_telephony` | Path to clips directory |
| `--api-url URL` | *(none)* | ALD API endpoint — omit for dummy/simulation mode |
| `--tag TAG` | `dummy` (sim) / `""` (real) | Label appended to the run directory name |
| `-j / --workers N` | `10` | Max concurrent API requests |
| `--runs-dir DIR` | `./runs` | Root directory where all run folders are created |
| `--serve` | off | Start HTTP server after evaluation |
| `--port N` | `8080` | HTTP server port (used with `--serve`) |

## Real API contract

The script POSTs `multipart/form-data` with field `audio` (WAV bytes). Expected JSON response:

```json
{
  "detected_language": "bn",
  "confidence": 0.92,
  "all_scores": { "bn": 0.92, "ml": 0.03, "ta": 0.02, "...": "..." }
}
```

`all_scores` is optional. Update `_real_detect()` in `evaluate.py` if the API shape differs.

## Outputs

| File | Contents |
|---|---|
| `predictions.csv` | Per-file: ground truth, predicted, confidence, latency, correct flag |
| `metrics.json` | Overall accuracy, per-language P/R/F1, per-duration accuracy, confusion matrix |
| `report_data.json` | Combined data consumed by `report.html` |
