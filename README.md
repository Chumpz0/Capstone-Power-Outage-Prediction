# Storm-to-Outage

DSCI 4350 capstone · Georgia State University · Fall 2026.

Estimate the probability that a county reaches **5% of customers out** during
the 24 hourly bins beginning at its first storm-report hour. The data support
retrospective storm/outage research; whole-episode reports stand in for forecasts.
Base-model train: 2017–2023 · development/calibration: 2024 · final test: 2025.
The frozen 2025 evaluation is complete: AP **0.525606** versus **0.398987** for
the reference; all three reference-relative criteria passed. See
[final results](docs/evaluation.md#final-2025-results-2026-11-25).

## Start here

1. [Data audit](notebooks/01_data_audit.ipynb): input availability, schema and denominators.
2. [EDA](notebooks/02_eda.ipynb): event-level and modeling-dataset findings, displayed inline.
3. [Modeling](notebooks/03_modeling.ipynb): historical baselines/candidate and run links. The latest selected model and final evaluation are documented in the evaluation guide.

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

The default `boosting` command reproduces the historical 100-tree/depth-3
candidate. The latest nested-CV selection uses **300 trees, learning rate 0.2,
depth 2**, fitted on 2017-2023. To reproduce those fixed base-model settings:

```powershell
python -m outage_prediction.train boosting --n-estimators 300 --learning-rate 0.2 --max-depth 2 --offline --output-dir models/selected-1
```

### Frozen 2025 evaluation

Use the existing saved nested run; do not retrain or rebuild the dataset for final
scoring. Run these commands in order, with new output directories:

```powershell
python -m outage_prediction.final_evaluation freeze --run-dir models/nested-20261124-215019 --offline --output-dir models/final-20261125-frozen
python -m outage_prediction.final_evaluation benchmark --frozen-dir models/final-20261125-frozen --offline --output-dir models/final-20261125-benchmark
python -m outage_prediction.final_evaluation score --frozen-dir models/final-20261125-frozen --allow-holdout --offline --output-dir models/final-20261125-test
```

Freeze compares a sigmoid calibrator fitted on January-June 2024 with raw
probabilities on July-December 2024, purging shared episodes from calibration
fitting. It retains calibration only for better Brier without worse AP/top-10%
recall. The base model is never refit. It saves the chosen predictor, frozen
reference, source snapshot, checksummed manifest, and all 32,098 keyed 2025
predictions **without reading test labels**. Benchmark measures latency and
whole-process memory in a fresh CLI process. Score explicitly opens labels,
requires exact key alignment and unchanged frozen artifacts/source, and saves
metrics, episode-bootstrap uncertainty, diagnostics, plots, and provenance.

Ordinary training and notebooks still cannot load test outcomes. Raw labels and
row predictions remain local; no final-evaluation command uploads anything.
Results describe held-out retrospective performance. After final scoring, do not
use 2025 to select models, features, calibration, or thresholds. See the
[evaluation protocol](docs/evaluation.md#frozen-2025-evaluation-protocol).

`--tune` samples boosting configurations without replacement using seed 42 and
selects the highest 2024 validation average precision (first trial wins ties).
The default space is 50/100/200/300 estimators, learning rates
0.01/0.03/0.1/0.2, and depths 1/2/3; `--n-trials` is capped at the space size.
Every candidate fits on 2017-2023 only. The saved pipeline is the winning fitted
candidate, with all trial scores in `hyperparameter_trials.csv` and search settings
in `config.json`. Validation scores are model-selection estimates; tuning never
reads 2025 outcomes. Fixed parameter flags cannot be combined with `--tune`.
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

Compare saved outer predictions with the reference without refitting any models:

```powershell
python -u -m outage_prediction.compare_cv --run-dir models/nested-1 --offline --n-boot 1000 --output-dir reports/generated/nested-reference-1
```

This rebuilds the reference within each outer training fold, checks saved row keys
and outcomes, and reports AP, Brier, and top-10% recall with paired episode-bootstrap
intervals for AP gain. Output contains aggregate metrics only. See the
[recorded comparison](docs/evaluation.md#outer-fold-reference-comparison) for the real-data results.

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
