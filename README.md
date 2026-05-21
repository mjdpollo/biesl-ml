# biesl-ml

Machine-learning experiments on multimodal physiological signals captured from a wearable device.

**Task.** Per-window 4-class classification — `rest` / `meditation` / `stress` / `recovery`.
**Features.** The eight parameters defined in [`features.pdf`](features.pdf), nothing else.
**Headline result.** XGBoost on those 8 features under leave-one-recording-out reaches **macro-F1 0.723**. Full numbers and confusion matrices in [`report.md`](report.md).

## Dataset

The raw recordings live in this Google Drive folder (shared link, view access):

> **https://drive.google.com/drive/u/0/folders/11epSBil0cIWSKvtShCp86gCrUKEOjxkn**

To fetch fresh into `data/`:

```bash
mkdir -p data/_old && mv data/*.txt data/_old/ 2>/dev/null
cd data && uv run --group dev gdown --folder \
    "https://drive.google.com/drive/folders/11epSBil0cIWSKvtShCp86gCrUKEOjxkn"
cd .. && mv "data/Stress test data/"*.txt data/
```

`data/` is gitignored. The folder currently contains 9 `*.txt` recordings (`mta`, `mta2` subjects, sessions 5-17 and 5-19) plus an `AAA read me.docx` (renamed to `README_dataset.docx` on disk).

### Signal channels (per recording)

Each recording is a tab-separated text file with four time-aligned channel pairs. Column **indices** are stable across the schema variants in the dataset (e.g. `time_ecg / data_ecg` vs `ads_time_2 / ads_ch2_data`); [`src/io.py`](src/io.py) reads by index, not by header name:

| Channel    | Index pair | Approx. rate | Used by features.pdf for                          |
| ---------- | ---------- | ------------ | -------------------------------------------------- |
| Microphone | 0, 1       | ~2000 Hz     | `csi` (Shannon-energy envelope → S1/S2 ratio)      |
| Breathing  | 2, 3       | ~500 Hz      | `rr`, `rrv` (slope-based peak detector)            |
| ECG        | 4, 5       | ~500 Hz      | `hr`, `hrv_rmssd`, `hrv_lf`, `hrv_hf`, `hrv_lf_hf` |
| Skin temp  | 6, 7       | ~1 Hz        | **not used** (excluded per features.pdf)           |

Each channel has its own time vector — they are not row-aligned and must be resampled to a common grid before training. The pipeline does this internally.

### Protocol

Every recording is `rest → stress → recovery`:

* `rest`: 0–5 min (300 s)
* `stress`: 5 min – (5 + stressor duration). For `medi` recordings the stressor is 5 min, ending at 10 min. For `pla` (plank) recordings the duration is parsed from the filename (e.g. `pla_1'40` = 1 min 40 s).
* `recovery`: end of stress phase → end of file (typically ~15 min total).

**Boundary policy.** Per patient request, windows that touch the 5-min or 10-min protocol transitions are excluded from training and evaluation. See `src.features._window_touches_boundary`.

## Repository layout

```
biesl-ml/
├── data/                   # raw recordings (gitignored)
│   └── _old/               # previous dataset, kept for reference
├── figures/                # tracked plots (confusion matrices, etc.)
├── notebooks/              # exploratory analysis
├── scripts/                # one-off runners (confusion matrices, etc.)
├── src/                    # pipeline code (loaders, preprocessing, models)
├── features.pdf            # spec for the 8 features used by the project
├── report.md               # teammate-facing report with embedded figures
├── confusion-matrices.md   # 8 confusion matrices (3 classical × 2 protocols + 1D-CNN × 2)
├── README.md
└── pyproject.toml
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Dependencies are declared in `pyproject.toml`; the lockfile is `uv.lock`.

```bash
uv sync                    # install runtime deps (numpy, pandas, scipy, sklearn, torch, etc.)
uv sync --group notebook   # + JupyterLab + ipykernel + ipywidgets
uv sync --group dev        # + gdown (for re-downloading the dataset)
```

## End-to-end commands

```bash
# Classical models (KNN / RandomForest / XGBoost), LORO + 70:15:15 random split
uv run python -m src.local_eval

# 1D-CNN, LORO + 70:15:15 random split (RTX 5090 in ~1 min, CPU in ~10)
uv run python -m src.dl_train

# Regenerate confusion matrices + heatmap PNGs in figures/confusion/
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

Output JSONs land in `outputs/` (gitignored):

| File                                 | Contents                                              |
| ------------------------------------ | ----------------------------------------------------- |
| `outputs/local_loro.json`            | classical, per-recording LORO + summary               |
| `outputs/local_randomsplit.json`     | classical, 5-seed 70:15:15 random split + summary     |
| `outputs/dl_local_loro.json`         | 1D-CNN, per-recording LORO + summary                  |
| `outputs/dl_local_randomsplit.json`  | 1D-CNN, 5-seed 70:15:15 random split + summary        |

## Where to look first

* [`report.md`](report.md) — the teammate-facing report (tables, all 8 confusion-matrix PNGs).
* [`confusion-matrices.md`](confusion-matrices.md) — confusion matrices as markdown tables.
* [`src/features.py`](src/features.py) — the 8-feature implementation that matches `features.pdf`.
* [`src/preprocess.py`](src/preprocess.py) — filters and peak detectors per the spec.
* [`src/local_eval.py`](src/local_eval.py) / [`src/dl_train.py`](src/dl_train.py) — evaluation drivers.
