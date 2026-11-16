# Evaluation policy

Train on 2017–2023, develop on validation 2024, and evaluate the frozen model
once on 2025 outcomes. Ordinary loaders and notebooks never resolve test labels.
Dataset/label definitions are in [dataset.md](dataset.md).

## Reference and success criteria

The reference is `clip(family_rate × hist_outage_rate / train_positive_rate, 0, 1)`.
Family rate is the highest training outage rate among families present in a row.
A row with no supported training family raises an error rather than producing NaN.
Historical 2024 reference metrics are AP 0.469056, Brier 0.111560, and top-10%
recall 32.5781%. Evaluate model/reference on identical rows at full precision:

1. Average precision must exceed the reference.
2. Brier score must be no higher than the reference.
3. Recall among the top 10% of rows must be no lower than the reference.

These reference-relative criteria replaced earlier arbitrary numerical targets on
2026-10-27, after initial logistic validation results existed. They are not a
pre-training commitment. The old AP ≥ 0.52, Brier skill ≥ 0.25 and year-to-year
AP-tolerance targets no longer apply.

## Metric contracts and diagnostics

“PR-AUC” in historical outputs means scikit-learn average precision, not trapezoidal
integration. Brier skill is `1 − Brier / [p_eval × (1 − p_eval)]`; evaluation
prevalence normalizes the score and is not a deployable learned prediction.
Top-10% metrics flag `ceil(0.10 × rows)` rows and preserve row order when scores tie.
Probability arrays are positional; Series indexes must match the evaluated rows.
Required labels, keys, split years and finite [0,1] scores are validated explicitly.

Report calibration deciles, overall calibration gap, storm-family/FEMA summaries,
and yearly scores. These are diagnostics, not additional hidden pass thresholds.
An aggregate pass does not imply that every subgroup improves or probabilities
are adequately calibrated. Initial logistic AP was 0.363576 and failed the three
comparisons; selected boosting AP was 0.588047 with Brier 0.098849.

## Uncertainty, persistence, and uploads

`evaluate(..., bootstrap=True)` explicitly requests the paired 95% percentile
interval for AP gain, resampling whole episode IDs: 1,000 draws, seed 42.
Supplying reference scores alone never starts bootstrapping. An interval above
zero supports an improvement claim; it remains separate from the three checks.
Historical feature-comparison intervals are exploratory and unadjusted for multiple
comparisons. They condition on fitted models and do not include retraining uncertainty.

Notebook training returns fitted results inline. Explicitly saving a run retains
the pipeline, keyed validation predictions, metrics, subgroup/calibration tables,
charts and actual-source metadata. Uploads are opt-in and preflight required files.
They exclude raw data, row predictions and test files. Models/configuration and
aggregate diagnostics are uploaded. Historical model-comparison artifacts created
before the refactor may attach the last model's metrics file; scalar summaries
were correct. No remote repair was performed.

See the [modeling notebook](../notebooks/03_modeling.ipynb) for historical comparisons
and recorded run links. Further model changes use validation only; the final
test evaluation compares the frozen model and training-derived reference on the
same rows, without tuning on the result.
