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
python -m outage_prediction.train boosting --tune --n-trials 12 --offline --output-dir models/tuned-1
python -m unittest discover -s tests -v
```

Model commands save reproducibility artifacts in a new run directory. Notebook
fits do not save files. Add `--bootstrap` for episode-level uncertainty; uploads
are off by default. To upload a completed real-data run explicitly:

```powershell
python -m outage_prediction.tracking models/boosting-1
```

`--tune` samples boosting configurations without replacement using seed 42 and
selects the highest 2024 validation average precision (first trial wins ties).
The default space is 50/100/200/300 estimators, learning rates
0.01/0.03/0.1/0.2, and depths 1/2/3; `--n-trials` is capped at the space size.
Every candidate fits on 2017-2023 only. The saved pipeline is the winning fitted
candidate, with all trial scores in `hyperparameter_trials.csv` and search settings
in `config.json`. Validation scores are model-selection estimates; 2025 stays
sealed. Fixed parameter flags cannot be combined with `--tune`.
`--smoke-test --tune --n-trials 2` exercises a small synthetic search.
In notebooks, `tune_model(train, validation, n_trials=12)` returns the same
`ModelResult` as `fit_model` without saving; import it from `outage_prediction.train`.

For nested cross-validation with expanding-year folds:

```powershell
python -u -m outage_prediction.nested_cv --offline --n-trials 12 --outer-folds 3 --inner-folds 3 --output-dir models/nested-1
```

Defaults score outer years 2021/2022/2023, tuning each model on the last three
available earlier years. Each inner/outer training window expands from 2017;
episodes shared with its held-out year are purged from training. County histories
are rebuilt using only that fold's training outcomes and priors, preserving the
released leave-one-training-year-out definition. Training-row histories remain
retrospective within that window. After outer scoring, a separate inner search
on all 2017-2023 selects the final configuration, which is fitted on those years
and evaluated on 2024. 2024 is excluded from nested selection, though it has been
used in previous project development. 2025 is never loaded.

The run saves `nested_cv_trials.csv`, `nested_cv_outer_folds.csv`, keyed outer
predictions in `nested_cv_predictions.csv`, and the final model/2024 diagnostics.
`config.json` includes outer mean, standard deviation, and pooled scores.
Progress is printed as JSON after each candidate and outer fold. The defaults
require 148 model fits (12 candidates across 3 inner folds in each of 3 outer
folds, plus 3 outer fits, a final 36-fit search, and a final model fit).

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
