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

Hyperparameter optimization (`train --tune`) selects boosting configurations by
2024 validation average precision using reproducible random sampling. Each
candidate's imputer and classifier fit only on 2017-2023. No random row split,
test labels, or refit on validation is used. The search retains all trial scores
and the winning fitted candidate. Report the selected validation metrics as
development estimates because they are used for selection. Optional bootstrap
intervals are computed only for the winner and do not account for search uncertainty.

`python -m outage_prediction.nested_cv` provides a separate nested expanding-year
experiment inside 2017-2023. By default, outer held-out years are 2021, 2022, and
2023; candidate selection maximizes the unweighted mean AP across three earlier
inner held-out years. All preprocessing is fit in each training fold. Released
county histories are discarded and recomputed from fold-training labels/peaks
and fold-training priors, with each training row's own year excluded from county
sums. Shared storm episodes are purged from fold training. Training-row histories
retain the retrospective within-window definition; this does not establish a
fully chronological forecasting dataset. Outer scores estimate the selection
procedure; their standard deviation describes variation across years, not a
confidence interval. Pooled AP is also reported and can differ from mean yearly AP.
The final inner search uses all training years and leaves 2024 out of selection.
2024 evaluation is still a development check given prior project use. The sealed
2025 outcomes remain reserved for the final frozen-model evaluation.

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

## Optimization results: 2026-11-24

Both real-data experiments used 208,482 training rows (2017-2023), 31,064
validation rows (2024), seed 42, and 12 sampled configurations. The direct
validation search and the final inner search in nested CV selected the same
configuration: **300 estimators, learning rate 0.2, maximum depth 2**. Their saved
models produced identical 2024 metrics:

| 2024 metric | Selected model | Training-derived reference |
|---|---:|---:|
| Average precision | 0.590379 | 0.469056 |
| Brier score (lower is better) | 0.098250 | 0.111560 |
| Recall in highest-risk 10% | 39.94% | 32.58% |

All three reference-relative criteria passed. The direct search used these 2024
scores for selection; the nested experiment did not. 2024 has been used in prior
project development, so neither score is a fresh final holdout result.

Nested CV completed 148 fits. All three outer searches chose 300 estimators,
learning rate 0.1, and depth 1 using only their earlier inner folds:

| Outer year | Training years | Held-out rows | Purged training rows | AP | Brier |
|---|---|---:|---:|---:|---:|
| 2021 | 2017-2020 | 27,711 | 0 | 0.514840 | 0.102495 |
| 2022 | 2017-2021 | 34,681 | 3 | 0.481226 | 0.096059 |
| 2023 | 2017-2022 | 30,768 | 3 | 0.565612 | 0.104724 |

Mean outer AP was **0.520559**, with across-year population standard deviation
**0.034687**; mean outer Brier was **0.101093**. Pooling the 93,160 outer predictions
gave AP **0.522248** and Brier **0.100835**. The final inner search's best mean AP
was 0.521195. Outer scores describe the selection procedure trained on smaller,
earlier windows; they are distinct from the final model's 2024 evaluation.
Fold-local histories retain the retrospective training-prior limitation described
above. No 2025 features or outcomes were loaded.

Post-run validation passed all 20 unit/integration tests, Python compilation,
all three notebook schema checks, source-hash matching, saved-model prediction
reproduction, row alignment, independent metric/reference recomputation, inner
selection checks, and outer/pooled score recomputation. Notebooks were schema
validated, not executed. Bootstrap uncertainty was not requested for these runs.

Aggregate evidence: [results and provenance](../reports/optimization/2026-11-24/results.json),
[validation-search trials](../reports/optimization/2026-11-24/hyperparameter_trials.csv),
[outer-fold scores](../reports/optimization/2026-11-24/nested_cv_outer_folds.csv), and
[inner-fold trials](../reports/optimization/2026-11-24/nested_cv_trials.csv).
Models, plots, and row predictions remain in ignored local run directories
`models/tuned-20261124-214757` and `models/nested-20261124-215019`.
