# biesl-ml

Machine-learning experiments on multimodal physiological signals captured from a wearable device.

**Task.** Per-window **4-class** classification — `rest` / `meditation` / `plank` / `math`. The post-stressor `recovery` phase is dropped.
**Features (9).** `csi, hr, hrv_rmssd, sd1, sd2, sd1_sd2, ss, rr, rrv`. LF / HF / LF-HF were replaced with Poincaré non-linear features after the window length dropped below what Welch reliably resolves.
**Windowing.** Anchor-based on a 2-s slide; each feature is computed on its own centered window (HR 10 s; RR/CSI 40 s; RMSSD / Poincaré / RRV 60 s). Asymmetric −10 / +30 s buffer around the 5-min cue.
**Headline result.** 1D-CNN reaches **pooled-LORO macro-F1 0.767** (neurokit BR) / 0.756 (sliding BR); XGBoost reaches 0.690 / **0.718** under the same swap. Reports live under [`report/`](report/):

- [`report/ml-report.md`](report/ml-report.md) — neurokit BR detector (default). Model numbers, per-class F1, confusion matrices.
- [`report/ml-report-sliding.md`](report/ml-report-sliding.md) — sliding BR detector. Head-to-head deltas vs neurokit.
- [`report/poincare-report.md`](report/poincare-report.md) — Poincaré scatter plots per recording × phase, plus aggregate per-stressor figures (detector-invariant — applies to both runs).

## Dataset

The raw recordings live in three Google Drive folders, one per stressor type (shared links, view access):

| Stressor | Google Drive folder |
|---|---|
| medi  | https://drive.google.com/drive/folders/12UDbV2YKf7ox5Ccxuxpk49MtxlrKPDcp |
| plank | https://drive.google.com/drive/folders/1cQ5k3jATUBV6Sz34C-GOnKGOOiZQFwPe |
| math  | https://drive.google.com/drive/folders/1LYBvZgsehhhe6M0W0jyD2-rxRrBTRhYE |

To fetch fresh into `data/` (move old aside, download all three, flatten):

```bash
mkdir -p data/_old && mv data/*.txt data/_old/ 2>/dev/null
mkdir -p data/_dl && cd data/_dl
for id in 12UDbV2YKf7ox5Ccxuxpk49MtxlrKPDcp \
          1cQ5k3jATUBV6Sz34C-GOnKGOOiZQFwPe \
          1LYBvZgsehhhe6M0W0jyD2-rxRrBTRhYE; do
    uv run --group dev gdown --folder "https://drive.google.com/drive/folders/$id"
done
cd .. && mv _dl/*/*.txt . && rm -rf _dl
```

`data/` is gitignored. It currently holds **31 `*.txt` recordings** across 7 subjects
(`mta`, `mta2`, `nvt`, `ntv`, `nva`, `oyj`, `smj`) and three stressors. Filenames ending in
`_posiECG` flag recordings whose ECG R-peaks deflect **positive** (the default device is negative);
`src/io.py` parses this into `Recording.ecg_polarity`.

### Signal channels (per recording)

Each recording is a tab-separated text file with four time-aligned channel pairs. Column **indices** are stable across the schema variants in the dataset (e.g. `time_ecg / data_ecg` vs `ads_time_2 / ads_ch2_data`); [`src/io.py`](src/io.py) reads by index, not by header name:

| Channel    | Index pair | Approx. rate | Used by features.pdf for                           |
| ---------- | ---------- | ------------ | -------------------------------------------------- |
| Microphone | 0, 1       | ~2000 Hz     | `csi` (Shannon-energy envelope → S1/S2 ratio)      |
| Breathing  | 2, 3       | ~500 Hz      | `rr`, `rrv` (slope-based peak detector)            |
| ECG        | 4, 5       | ~500 Hz      | `hr`, `hrv_rmssd`, Poincaré `sd1` / `sd2` / `sd1_sd2` / `ss` |
| Skin temp  | 6, 7       | ~1 Hz        | **not used** (excluded per features.pdf)           |

Each channel has its own time vector — they are not row-aligned and must be resampled to a common grid before training. The pipeline does this internally.

### Protocol

Every recording is `rest → stress → recovery`:

- `rest`: 0–5 min (300 s)
- `stress`: 5 min – (5 + stressor duration). For `medi` recordings the stressor is 5 min, ending at 10 min. For `pla` (plank) recordings the duration is parsed from the filename (e.g. `pla_1'40` = 1 min 40 s).
- `recovery`: end of stress phase → end of file (typically ~15 min total).

**Boundary policy.** Per patient request, windows that touch the 5-min or 10-min protocol transitions are excluded from training and evaluation. See `src.features._window_touches_boundary`.

## Repository layout

```
biesl-ml/
├── data/                   # raw recordings (gitignored)
│   ├── _excluded/          # full-file exclusions (hardware / duplicates)
│   └── _old_pre_*/         # previous dataset snapshots
├── figures/                # tracked plots (confusion matrices, Poincaré, BR detector compare)
├── notebooks/              # exploratory analysis
├── scripts/                # runners (run_split_reports.py, dump_preprocessed_nn.py, plot_poincare.py, …)
├── src/                    # pipeline code (loaders, preprocessing, features, models)
├── report/                 # self-contained ML + Poincaré reports with embedded figures
├── features.pdf            # original 8-feature spec (superseded — see ml-report.md)
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
# All 4 models (KNN / RF / XGBoost / 1D-CNN), 4-class, LORO + 70:15:15 random split.
# Default BR detector is neurokit; set BR_PEAK_METHOD=sliding|global to swap.
BR_PEAK_METHOD=neurokit uv run python scripts/run_split_reports.py

# BR-detector ablation (runs all 3 detectors back-to-back)
uv run python scripts/run_detector_compare.py

# Dump per-recording NN intervals for Poincaré analysis
uv run python scripts/dump_preprocessed_nn.py

# Render Poincaré scatter PNGs into figures/poincare/
uv run python scripts/plot_poincare.py
```

Outputs:

| File                                          | Contents                                              |
| --------------------------------------------- | ----------------------------------------------------- |
| `outputs/split_reports.json`                  | All numbers (per-class F1, confusion, per-fold)       |
| `outputs/preprocessed_nn.json`                | Per-recording × phase NN-interval / BR-interval pools |
| `figures/with_math/confusion/*.png`           | Confusion-matrix heatmaps (4 models × 2 protocols)    |
| `figures/poincare/*.png`                      | Poincaré scatter plots (per-recording + aggregate)    |

## Where to look first

- [`report/ml-report.md`](report/ml-report.md) — 4-class numbers, per-class F1, confusion matrices.
- [`report/poincare-report.md`](report/poincare-report.md) — Poincaré per recording × phase, with SD1 / SD2 / SS values.
- [`src/features.py`](src/features.py) — 9 features + anchor-based per-feature windowing.
- [`src/preprocess.py`](src/preprocess.py) — filters and peak detectors.
- [`src/raw_windows.py`](src/raw_windows.py) — CNN raw windows (40 s, 3 ch).
