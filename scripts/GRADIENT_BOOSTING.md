# Gradient boosting implementation handoff

`07_gradient_boosting.py` is a separate initial classifier implementation. It leaves
the baseline script and shared evaluation unchanged. It uses scikit-learn already
in requirements; joblib is installed with scikit-learn. No new dependency is needed.

The explicit input allowlist matches logistic regression: `month`, `local_hour`,
`pre_peak_24h`, `pct_out_at_start`. Step 04 defines the last two using hours before
the storm; missing pre-storm observations remain missing until imputation. IDs,
year, target, and all `retro_` outcomes are excluded from model inputs. The wider
storm/history feature set is intentionally left for the teammate's experiments.

The pipeline fits median imputation and missing indicators on 2017–2023 only.
Validation must contain only 2024. There is no random train/validation split,
early stopping, automatic tuning, class balancing, or threshold selection. The
script opens only `train.parquet` and `validation.parquet`, preferring local files
and otherwise using the same private Hugging Face location as step 05. It never
opens 2025 features or labels. Year, label, key, and feature checks fail before fit.

From the repository root, using the project environment:

```bash
python scripts/07_gradient_boosting.py --help
python scripts/07_gradient_boosting.py --smoke-test
```

The smoke test needs no dataset access. It trains five trees on synthetic rows,
checks missing-value handling, finite probabilities, saved-model round trip, and
integration with `src.evaluation.evaluate`. Its metrics are meaningless for model
quality. Outputs go to ignored `work/gradient_boosting/smoke/`. Repeated runs must
use a new location, e.g. `--output-dir work/gradient_boosting/smoke-2`.

For the teammate to run later (not required for implementation acceptance):

```bash
python scripts/07_gradient_boosting.py --output-dir work/gradient_boosting/run-1
python scripts/07_gradient_boosting.py --evaluate --n-estimators 200 --learning-rate 0.05 --max-depth 2 --output-dir work/gradient_boosting/run-2
```

Private remote inputs require dataset permission and `hf auth login`. On Windows,
an activated environment's `python` can be replaced with `.venv\Scripts\python.exe`.
Relative output paths resolve from the working directory. Existing output folders
are refused to protect earlier runs.

Each run saves the fitted preprocessing/classifier pipeline (`model.joblib`),
validation probabilities with county/episode keys (`validation_predictions.csv`),
and features, sources, parameters, row counts, and split years (`config.json`).
Only load joblib files you trust. Optional `--evaluate` saves shared metrics and
calibration/family/region/year tables, including comparison to the training-derived
reference rule. Like the baseline's default evaluation it omits the optional
episode bootstrap. No W&B upload occurs.

The implementation owner supplies this script and smoke checks. The teammate owns
full training, feature experiments, tuning on 2024, calibration, threshold choice,
bootstrap evaluation, and interpretation. Smoke success is not evidence of better
performance. Keep 2025 sealed until the team's final approved evaluation.
