# biesl-ml

Machine learning experiments on multimodal physiological signals captured from a wearable device.

Data link : https://drive.google.com/drive/u/1/folders/11epSBil0cIWSKvtShCp86gCrUKEOjxkn

## Signals

Each recording is a tab-separated text file with four time-aligned channel pairs
(`time_*_s` in seconds, sampled independently per channel):

| Channel    | Time column          | Value column           | Approx. rate |
| ---------- | -------------------- | ---------------------- | ------------ |
| Microphone | `time_microphone_s`  | `microphone_data`      | ~2000 Hz     |
| Breathing  | `time_br_s`          | `br_data`              | ~500 Hz      |
| ECG        | `time_ecg_s`         | `ecg_data`             | ~500 Hz      |
| Skin temp. | `time_temperature_s` | `temperature_object_C` | ~1 Hz        |

Each channel has its own time vector — they are **not** row-aligned and must be
resampled or windowed onto a common grid before training.

## Repository layout

```
biesl-ml/
├── data/         # raw recordings (gitignored)
├── notebooks/    # exploratory analysis and model training
├── src/          # reusable preprocessing / model code
├── .gitignore
└── README.md
```

`data/` is excluded from version control — drop the raw `*-medi.txt` files in
there locally. The current dataset contains:

- `mta-5-8-medi.txt`
- `nvt-5-8-medi.txt`
- `nvt-5-15-medi.txt`

Filename convention: `{subject}-{M}-{D}-medi.txt`.

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. Dependencies are declared in `pyproject.toml`; the lockfile is
`uv.lock`.

```bash
# Install runtime deps (numpy, pandas, scipy, scikit-learn, matplotlib, torch)
uv sync

# Or include the notebook tooling (jupyterlab, ipykernel, ipywidgets)
uv sync --group notebook
```

A Jupyter kernel named **Python (biesl-ml)** is registered against this
environment. Re-register it any time after recreating `.venv`:

```bash
uv run --group notebook python -m ipykernel install --user \
    --name biesl-ml --display-name "Python (biesl-ml)"
```

Launch JupyterLab:

```bash
uv run --group notebook jupyter lab
```

Add new packages with `uv add <pkg>` (runtime) or `uv add --group notebook <pkg>`
(notebook-only).

Notebooks live in `notebooks/`; shared loaders and feature code go in `src/`.

## Loading a recording

Column header names vary across files (some have trailing spaces, a typo
`rime_br`, or no `_s` suffix), so `src.io.load_recording` reads by column
**index** (0,2,4,6 = time columns; 1,3,5,7 = value columns).

```python
from src.io import load_recording
rec = load_recording("data/mta-5-17-pla-1'26.csv")
print(rec.subject, rec.stressor, rec.plank_seconds)  # mta pla 86.0
print(rec.channels["ecg"].shape)                     # (2, n_samples)
```

## KNN pipeline

End-to-end (load → filter → peak detect → window → features → KNN with
subject-grouped CV):

```bash
uv run python -m src.pipeline   # writes outputs/
uv run python -m src.plots      # writes signal + confusion diagnostics
```

Outputs land in `outputs/` (gitignored):

| File                  | Contents                                                       |
| --------------------- | -------------------------------------------------------------- |
| `features.csv`        | per-window feature table (meta + ~28 features)                 |
| `loso_results.json`   | per-fold metrics + aggregated confusion matrix                 |
| `confusion_loso.png`  | aggregated confusion matrix figure                             |
| `signals_*.png`       | raw vs filtered signal + detected peaks per recording          |
| `knn_model.joblib`    | final pipeline fit on all data                                 |
| `feature_names.joblib`| feature column order corresponding to the model                |

### Preprocessing

| Signal           | Resample fs | Filter                       | Peaks                          |
| ---------------- | ----------- | ---------------------------- | ------------------------------ |
| ECG              | 250 Hz      | BP 0.5–40 Hz + 60 Hz notch   | Pan–Tompkins (neurokit2) + kubios fix |
| BR               | 25 Hz       | BP 0.1–0.5 Hz + Savitzky–Golay | neurokit2 `rsp_peaks` (biosppy) |
| CPS (mic) — card | 100 Hz      | BP 0.8–3 Hz                  | `find_peaks` on bandpassed     |
| CPS (mic) — resp | 100 Hz      | BP 0.1–0.5 Hz                | `find_peaks` on bandpassed     |
| Skin temp        | ~1 Hz       | none                         | mean / std / slope per window  |

All bandpasses use second-order-section form (`sosfiltfilt`) — the
`b,a / filtfilt` form is numerically unstable at the very low normalized
cutoffs needed for the BR / CPS-resp bands.

### Labels

Each recording has a known `rest → stress → recovery` structure. Phase
boundaries come from filename metadata (5 + 5 + 5 min for `medi`; 5 +
parsed-plank-duration + 5 min for `pla`). The model classifies these three
phases over 30 s windows with 50 % overlap.

### Subject-wise split

There are only two unique subjects (`mta`, `nvt`); evaluation is
LeaveOneGroupOut by subject. Inner CV for hyperparameter tuning groups by
recording when ≥3 training recordings are available, else falls back to
StratifiedKFold. With only two subjects, the within-subject inner-CV scores
are optimistic — the LOSO test scores are the honest generalization estimate.
