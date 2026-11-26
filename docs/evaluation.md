# Evaluation policy

Train on 2017–2023, develop on validation 2024, and evaluate the frozen model
once on 2025 outcomes. Ordinary loaders and notebooks never resolve test labels.
Dataset/label definitions are in [dataset.md](dataset.md).
The frozen 2025 evaluation was completed on **2026-11-25 (America/New_York)**;
see [final results](#final-2025-results-2026-11-25). 2025 is now an evaluated test
set, not an untouched resource for further model selection.

## Frozen 2025 evaluation protocol

The final predictor starts with `models/nested-20261124-215019/model.joblib`:
300 estimators, learning rate 0.2, depth 2, nineteen features, fitted only on
2017-2023. Before implementation, all original source/data hashes matched and its
saved 2024 predictions and metrics were reproduced to tolerance 1e-12.

`python -m outage_prediction.final_evaluation` provides three explicit actions:

1. **freeze** verifies the saved training/validation data and reproduces the raw
   2024 results. A sigmoid calibrator wraps the frozen base estimator and fits
   only January-June 2024; episodes also appearing in July-December are removed
   from calibration fitting. Raw and calibrated scores are compared on identical
   July-December rows. Calibration is selected only for strictly lower Brier and
   no AP/top-10% recall decrease beyond 1e-12. Otherwise raw probabilities remain
   selected. Infeasible/invalid calibration also retains raw with an explanation.
   The chosen calibrator is not refit on all 2024. This is development evidence,
   since 2024 has been used previously.
2. **benchmark**, in its own CLI process, measures thirty warmed predictions for
   one row and the complete 2024 batch. Median/p95 times, platform, versions,
   serialized size, current process RSS, and lifetime peak RSS are recorded.
   Whole-process memory includes imports and data; model bytes are storage, not RAM.
3. **score --allow-holdout** opens 2025 labels only after verifying checksums and
   the frozen source snapshot. It joins exact county/episode key sets one-to-one
   in feature-file order, requires all 32,098 expected rows, and rejects malformed
   outcomes, invalid probabilities, or episodes shared with development. It scores
   saved probabilities; it never fits or selects anything.

Freeze snapshots 2025 features/predictions, the selected predictor, original base
model/configuration, training-derived reference rates, excluded development
episode IDs, calibration membership/diagnostics, coverage sensitivity, code, and
hashes. Released train/test histories are independently rebuilt from 2017-2023
outcomes and checked without reading test outcomes. Full-training priors and
retrospective storm inputs remain unchanged.

The primary criteria remain AP > reference, Brier <= reference, and top-10%
recall >= reference on identical rows. Top-10% flags ceil(0.1 * rows), with stable
feature-file ordering for ties. AP means sklearn average precision. Brier skill
uses evaluation prevalence. A paired 95% AP-gain interval uses 1,000 whole-episode
bootstrap draws, seed 42; it is separate from the three point criteria and excludes
training/selection uncertainty. Family/FEMA and calibration diagnostics accompany
the final results without new pass thresholds.

Before unsealing, a 2024-only sensitivity comparison retains rows with at least
one county non-null reading on each label-window UTC date. It uses the same labels
and fixed predictions, does not establish continuous coverage, and does not alter
the primary 2025 population. Chronological histories, truly available storm
features, exact-minute labels, and stricter coverage definitions require a separate
dataset experiment; they are not retroactively introduced into this final test.

Use the commands in the [README](../README.md#frozen-2025-evaluation). All row-level
artifacts remain local. Final scoring preserves output directories and may be
reproduced for the same frozen predictor, but 2025 cannot guide subsequent model,
calibration, feature, or threshold selection. These results are held-out
retrospective performance, not validation of operational forecasting.

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
2025 outcomes were reserved until the final frozen-model evaluation below.

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

### Outer-fold reference comparison

The saved outer predictions were compared with `family_x_history`, rebuilt using
the same expanding-year, episode-purged training folds and fold-local histories.
All 93,160 rows and their targets were matched by county/episode/year keys. No
models were refit, no hyperparameters changed, and 2024/2025 were not loaded.

| Outer year | Model AP | Reference AP | AP gain | Paired 95% interval for gain |
|---|---:|---:|---:|---|
| 2021 | 0.514840 | 0.363946 | 0.150894 | [0.127269, 0.175479] |
| 2022 | 0.481226 | 0.364233 | 0.116994 | [0.100071, 0.132473] |
| 2023 | 0.565612 | 0.454875 | 0.110737 | [0.095391, 0.127347] |
| Pooled | 0.522248 | 0.397270 | 0.124978 | [0.114759, 0.135088] |

| Outer year | Model Brier | Reference Brier | Model top-10% recall | Reference top-10% recall |
|---|---:|---:|---:|---:|
| 2021 | 0.102495 | 0.117012 | 37.29% | 28.09% |
| 2022 | 0.096059 | 0.105460 | 37.69% | 30.99% |
| 2023 | 0.104724 | 0.114674 | 37.29% | 31.21% |
| Pooled | 0.100835 | 0.111939 | 37.79% | 30.57% |

The model passed all three reference-relative point criteria in every outer year
and in the pooled comparison. All AP-gain intervals were above zero. Intervals
use 1,000 paired whole-episode resamples, seed 42; the pooled bootstrap keeps
episodes spanning multiple years together. Year composition may vary across
pooled resamples. Pooled top-10% recall ranks all years together; yearly scores
allocate the risk budget separately within each year.

These results support improvement over the reference across the observed years.
Intervals condition on fitted models and exclude retraining/selection uncertainty;
yearwise intervals are exploratory and unadjusted for multiple comparisons.
They do not resolve the retrospective feature limitations or replace evaluation
of a frozen model on 2025. The full test suite passed 23 tests after adding this
comparison, including key alignment, target mismatch rejection, reproducibility,
and zero paired gain when model scores equal reference scores.

Aggregate evidence: [comparison table](../reports/optimization/2026-11-24/reference-comparison/reference_comparison.csv)
and [bootstrap settings and provenance](../reports/optimization/2026-11-24/reference-comparison/summary.json).

## Final 2025 results: 2026-11-25

The existing nested-CV base model was retained without refitting. Sigmoid
calibration was selected before test-label access: fitting used 19,864 rows from
January-June 2024; 26 earlier rows sharing episodes with July-December were
purged. Comparison used 11,174 July-December rows. Brier improved from 0.104849
to 0.104482, while AP remained 0.647390 and top-10% recall remained 38.87%.
The calibrator was not refit after comparison. These are development results.

The final bundle and all keyed 2025 predictions were frozen at
2026-11-26 03:33:40 UTC (November 25, 10:33:40 PM Eastern). Test labels were first
opened at 03:34:18 UTC, after checksums, source verification, synthetic tests,
and benchmarking. Every one of the **32,098** rows matched by county/episode;
there were 7,866 episodes and 4,961 positive rows (15.46%). No test rows were
dropped, and no model/calibration/threshold selection used the test outcomes.

| 2025 metric | Frozen calibrated predictor | Training-derived reference | Criterion |
|---|---:|---:|---|
| Average precision | **0.525606** | 0.398987 | Passed: higher |
| Brier score | **0.099618** | 0.111522 | Passed: no higher |
| Recall in highest-risk 10% | **38.22%** | 30.30% | Passed: no lower |
| Brier skill | 0.237637 | 0.146533 | Diagnostic |

ROC-AUC was 0.823758. Flagging 3,210 rows caught 1,896 positives, versus 1,503
for the reference: **393 additional positives** at the same illustrative risk
budget. Top-10% precision was 59.07%. This budget is not a demonstrated crew
capacity or an operational allocation policy.

AP gain was **0.126619**, with a paired whole-episode 95% bootstrap interval
**[0.110354, 0.143254]** (1,000 draws, seed 42). The interval supports improvement
over the reference for the frozen predictions; it excludes retraining and
selection uncertainty. AP exceeded the reference in all ten FEMA regions and
the six reported storm-family groups. These subgroup point estimates do not
establish significance or improvement in every metric or every storm family.

Mean predicted probability exceeded observed prevalence by 0.61 percentage
points; the worst absolute calibration-decile gap was 2.22 points. The
[calibration table](../reports/final_evaluation/2026-11-25/calibration_deciles.csv)
and plots accompany the aggregate evidence.

### Runtime and coverage checks

The selected predictor's fresh-process benchmark used Windows 11, Python 3.14.3,
scikit-learn 1.9.1, and the AMD processor identifier recorded in
[benchmark.json](../reports/final_evaluation/2026-11-25/benchmark.json). After one
warm-up per workload, thirty repetitions gave single-row median/p95 latency of
**1.29/1.90 ms**, and 31,064-row batch median/p95 latency of **38.70/46.32 ms**.
The predictor occupied 232,258 serialized bytes. Peak whole-process working set
was **225.7 MiB**, including imports, model and data; this is not isolated model
RAM or a guarantee on other hardware. Historical base-model fit time was 46.97 s.

The pre-test 2024 coverage sensitivity retained 23,679 rows and excluded 7,385
with no county non-null readings on at least one label-window date. On that
subset, prevalence was 20.71%, AP 0.613037, Brier 0.118740, and top-10% recall
34.34%. The changed population prevents interpreting this as a like-for-like
model improvement. It reinforces uncertainty in missing-as-zero labels. Primary
test eligibility and labels were unchanged; no 2025 coverage subset was selected.

### Evidence and interpretation

All **29 tests** passed. Python syntax, notebook schemas, frozen source/artifact
hashes, exact row alignment, and model prediction reproduction passed.
An independent NumPy implementation reproduced AP, Brier, top-10% metrics, and
the complete 1,000-draw episode-bootstrap interval within 1e-12. The independent
check also verified that freeze preceded label access and output hashes matched.

- [Results and validation summary](../reports/final_evaluation/2026-11-25/results.json)
- [Independent validation](../reports/final_evaluation/2026-11-25/independent_validation.json)
- [Frozen manifest](../reports/final_evaluation/2026-11-25/frozen_manifest.json)
- [Scoring provenance](../reports/final_evaluation/2026-11-25/provenance.json)
- [Calibration selection](../reports/final_evaluation/2026-11-25/calibration_selection.json)
- [Coverage sensitivity](../reports/final_evaluation/2026-11-25/coverage_sensitivity.json)

Local bundles are `models/final-20261125-frozen`,
`models/final-20261125-benchmark`, and `models/final-20261125-test`. Row-level
predictions and labels remain local; no uploads were performed. The older
modeling notebook is explicitly historical and does not read 2025 outcomes.

The final result supports the selected predictor over its reference on held-out
2025 retrospective storm episodes. It does not remove whole-episode feature
timing, training-history prior dependence, hour alignment, or coverage/denominator
limitations. Forecast-time feature availability and chronological histories
remain a separately versioned future experiment. 2025 must not guide additional
selection; further model development needs another evaluation design.
