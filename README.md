# Storm-to-Outage

DSCI 4350 capstone · Georgia State University · Fall 2026.

Estimate the probability that a county reaches **5% of customers out** during
the 24 hourly bins beginning at its first storm-report hour. The data support
retrospective storm/outage research; whole-episode reports stand in for forecasts.
Train: 2017–2023 · validation: 2024 · sealed test outcomes: 2025.

## Start here

1. [Data audit](notebooks/01_data_audit.ipynb): input availability, schema and denominators.
2. [EDA](notebooks/02_eda.ipynb): event-level and modeling-dataset findings, displayed inline.
3. [Modeling](notebooks/03_modeling.ipynb): baselines, selected boosting candidate and historical run links.

Definitions: [dataset](docs/dataset.md) · [evaluation](docs/evaluation.md).
Deliverable: [curated EDA report](reports/eda_report.md).

## Setup

```powershell
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe -e .
```

Select `.venv` as the notebook kernel. Activate it or replace `python` below
with `.venv\Scripts\python.exe`. Editable installation makes imports work
from notebooks or another working directory; runtime versions remain pinned in
`requirements.txt`.

Data is private in [DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data).
Use `hf auth login` with approved organization access. Loaders prefer local data,
then the configured fixed remote revision. Notebooks default to `offline=True`;
set it to `False` for pinned remote reads. Missing remote inputs fail without
silently reading latest. Old local data paths remain compatible during migration.

## Commands

```powershell
python -m outage_prediction.train --smoke-test --output-dir models/smoke-1
python -m outage_prediction.train baseline --offline --output-dir models/baseline-1
python -m outage_prediction.train boosting --offline --output-dir models/boosting-1
python -m unittest discover -s tests -v
```

Model commands save reproducibility artifacts in a new run directory. Notebook
fits do not save files. Add `--bootstrap` for episode-level uncertainty; uploads
are off by default. To upload a completed real-data run explicitly:

```powershell
python -m outage_prediction.tracking models/boosting-1
```

Rebuilding large data is separate from analysis and is not required when the
prepared release is available:

```powershell
python -m outage_prediction.dataset prepare --offline
python -m outage_prediction.features build --offline
```

Preparation replaces the validated interim generation and preserves the previous
one. Feature building similarly publishes the modeling tables. Test labels are
generated only with explicit `--write-holdout`; keep them sealed until final
evaluation. Routine report export requires an explicit `--export-dir` or
`export_results(..., directory=...)` call.

## Layout

```text
outage_prediction/    reusable preparation, features, analysis, models and plotting
notebooks/           three narrative notebooks; calculations import the package
docs/                dataset and evaluation definitions
reports/             one curated EDA deliverable and its referenced figures
data/                ignored raw/, external/, interim/, processed/, holdout/
models/              ignored model runs and historical artifacts
.cache/              ignored caches, migration backup and validation artifacts
```

History features retain other-training-year aggregation with full-training priors;
labels retain hour alignment. These limitations are explicit in the dataset guide.
A chronological forecast experiment requires a separate dataset revision.
