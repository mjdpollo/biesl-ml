# biesl-ml

Machine learning experiments on multimodal physiological signals captured from a wearable device.

## Signals

Each recording is a tab-separated text file with four time-aligned channel pairs
(`time_*_s` in seconds, sampled independently per channel):

| Channel       | Time column            | Value column             | Approx. rate |
|---------------|------------------------|--------------------------|--------------|
| Microphone    | `time_microphone_s`    | `microphone_data`        | ~2000 Hz     |
| Breathing     | `time_br_s`            | `br_data`                | ~500 Hz      |
| ECG           | `time_ecg_s`           | `ecg_data`               | ~500 Hz      |
| Skin temp.    | `time_temperature_s`   | `temperature_object_C`   | ~1 Hz        |

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

```python
import pandas as pd

df = pd.read_csv("data/mta-5-8-medi.txt", sep="\t")
ecg   = df[["time_ecg_s", "ecg_data"]].dropna()
br    = df[["time_br_s",  "br_data"]].dropna()
mic   = df[["time_microphone_s", "microphone_data"]].dropna()
temp  = df[["time_temperature_s", "temperature_object_C"]].dropna()
```
