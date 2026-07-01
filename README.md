# ALD Model Evaluation

Evaluates an Audio Language Detection (ALD) model against labelled telephony clips and produces per-run metrics, a visual HTML report, and exported PDF/PNG snapshots.

## Requirements

```bash
pip install scikit-learn aiohttp playwright
playwright install chromium
```

`scikit-learn` and `aiohttp` are needed for evaluation. `playwright` / `chromium` are needed for the automatic PDF and PNG export (skip with `--no-export` if unavailable).

## Directory structure

```
model-evaluation/
├── evaluate.py                        # evaluation pipeline + entry point
├── export.py                          # standalone PDF/PNG exporter
├── ald_client.py                      # async API client (real + dummy mode)
├── index.html                         # interactive report viewer
├── eval_clips_telephony/              # input data — never modified by scripts
│   ├── metadata.csv                   # ground truth: file, language, duration_sec, …
│   └── bn/  gu/  kn/  ml/  mr/  ta/  te/
│       └── 1s/  2s/  3s/  5s/
│           └── *.wav
└── runs/                              # created on first run
    ├── 2026-07-01_110755_v1/
    │   ├── run_info.json              # config snapshot: tag, api, workers, git hash
    │   ├── predictions.csv            # per-file: ground truth, predicted, confidence, latency
    │   ├── metrics.json               # accuracy, per-language P/R/F1, confusion matrix
    │   ├── report_data.json           # data consumed by index.html
    │   ├── report.pdf                 # exported report (landscape A3)
    │   └── report.png                 # full-page screenshot
    ├── 2026-07-01_093020_v2/
    │   └── …
    └── latest -> 2026-07-01_093020_v2   # symlink, always points to most recent run
```

## Run evaluation

```bash
# Dummy/simulation mode — no API needed, useful for testing the pipeline
python3 evaluate.py

# Real ALD API
python3 evaluate.py --api-url http://your-host/detect

# Named run for easy identification
python3 evaluate.py --api-url http://your-host/detect --tag v1.2

# Throttle concurrency and open the report in a browser when done
python3 evaluate.py --api-url http://your-host/detect --tag v1.2 -j 5 --serve

# Skip automatic PDF/PNG export
python3 evaluate.py --no-export
```

Each run creates a timestamped directory under `runs/` (e.g. `runs/2026-07-01_110755_v1`), updates the `runs/latest` symlink, and by default exports `report.pdf` and `report.png` into that directory.

The version tag is auto-incremented (`v1`, `v2`, …) when `--tag` is omitted.

## CLI reference — `evaluate.py`

| Argument | Default | Description |
|---|---|---|
| `input_dir` | `eval_clips_telephony` | Path to the clips directory |
| `--api-url URL` | *(none)* | ALD API endpoint — omit to run in dummy/simulation mode |
| `--tag TAG` | auto (`v1`, `v2`, …) | Label appended to the run directory name |
| `-j / --workers N` | `10` | Max concurrent API requests |
| `--runs-dir DIR` | `./runs` | Root directory for all run folders |
| `--no-export` | off | Skip automatic PDF/PNG export |
| `--serve` | off | Start HTTP server and open report after evaluation |
| `--port N` | `8080` | HTTP server port (used with `--serve`) |

## Export report manually

`export.py` re-exports PDF/PNG from any existing run without re-running evaluation.

```bash
# Export the latest run (PDF + PNG)
python3 export.py

# Export a specific run
python3 export.py runs/2026-07-01_110755_v1

# PDF only / PNG only
python3 export.py --format pdf
python3 export.py --format png
```

## View the interactive report

```bash
# Serve from the model-evaluation/ directory
python3 -m http.server 8080
```

Open **http://localhost:8080/** in a browser, then use the run selector to switch between runs.

Or run evaluation and serve in one step:

```bash
python3 evaluate.py --serve
```

## API contract

The evaluation script POSTs `multipart/form-data` with a single field `audio` (WAV bytes). Expected JSON response:

```json
{
  "detected_language": "bn",
  "confidence": 0.92,
  "all_scores": { "bn": 0.92, "ml": 0.03, "ta": 0.02 }
}
```

`all_scores` is optional. Update `_real_detect()` in `ald_client.py` if the response shape differs.

## Outputs

| File | Contents |
|---|---|
| `predictions.csv` | Per-file: ground truth, predicted, confidence, latency, correct flag |
| `metrics.json` | Overall accuracy, per-language P/R/F1, per-duration accuracy, confusion matrix |
| `report_data.json` | Combined payload consumed by `index.html` |
| `run_info.json` | Run metadata: tag, API URL, worker count, git hash, elapsed time |
| `report.pdf` | Landscape A3 PDF snapshot of the report |
| `report.png` | Full-page PNG screenshot of the report |
