# County preparation forecasting criteria

Use these rules to build and evaluate the baseline model. They define the targets, data splits and scores. Minimum recall and acceptable false-alert rates still need to be agreed.

## Prediction task and cohort

Issue a prediction for every study county at each UTC hour, separately for the next 6 and 12 hours. Predict (a) the uncapped maximum recorded quarter-hour customer count in `(t, t+h]`, and (b) its probability of being **strictly >5,000**. Sensitivity thresholds remain >1,000, >2,500 and >10,000. This measures future outage level, including an outage already in progress; it is not a forecast of new outage onset or incremental customers affected.

Use complete, unflagged outcomes for the primary labeled baseline cohort. Preserve incomplete and unknown decisions in the observation-coverage accounting. Partial observed exceedances remain a separate positive-evidence analysis, not complete regression outcomes or a pool to mix with complete negatives. Never use storms to select county-hour decisions. Predictions can be generated for all county-hours, but primary scoring must identify the subset with reliable labels. Do not call the unscored population safe or estimate its false-negative rate.

Report retention alongside scores: the training cohort retains only 7.70% / 4.74% of eligible 6/12-hour decisions. Outcome eligibility is retrospective and cannot be used as a prediction-time routing rule. A restrictive labeled cohort can support an initial comparison; it does not establish useful county-wide performance.

## Chronological splits and boundary exclusions

| Split | UTC decision origins | Role |
| --- | --- | --- |
| Development/train | 2014-11-01 04:00 through 2023 | EDA rules, feature choices and fitted model parameters |
| Validation | Calendar 2024 | Compare the frozen candidate models/latency scenarios; select probability thresholds and calibration if needed |
| Test | Calendar 2025 | One final evaluation after model and alert rules are frozen |

An origin belongs to a split only when `split_start <= t` and **`t + horizon < split_end`**. The exclusive end ensures the endpoint observation at midnight belongs to the following split. At each year-end split boundary this removes the last 6 or 12 origins per county: **3,204 / 6,408 county-hour origins**. Apply the same rule to the final 2026-01-01 data cutoff; do not fabricate outcomes after it. Internal missing observations are handled by completeness, not by extending a window. [Executable assignment](../src/forecast_criteria.py) implements and tests this rule.

Within development data, use forward chronological folds rather than randomly splitting overlapping rows. A simple proposed development sequence is fit through 2021, check 2022, then fit through 2022 and check 2023, purging future outcome windows at each fold boundary with the same rule. Fit preprocessing only on each fold's training subset. Historical predictor lookbacks may reach into earlier splits, provided they obey the as-of contract. The 12-hour purge prevents shared target observations across adjacent splits; it does not make neighboring windows statistically independent or eliminate storm persistence across a boundary.

Earlier descriptive notebooks already examined 2024/2025 data. Therefore 2025 is a **previously exposed historical evaluation set**, not a pristine blind test. The geography audit inspected 2025 storm mappings without reading outage outcomes. Keep this exposure disclosure in the handoff, freeze decisions before further holdout outcome analysis, and do not claim prospective validation. A genuinely untouched future evaluation requires later data, outside the current scope.

## Classification evaluation

Keep the outcome threshold (>5,000 customers) distinct from the model's alert probability threshold. Select the latter using validation data only and report it separately for each horizon.

| Measure | Definition / purpose |
| --- | --- |
| Severe-window recall | `TP / (TP + FN)`; primary priority because missed severe outages are more costly |
| Precision | `TP / (TP + FP)`; proportion of alerts supported by a severe labeled outcome |
| False alerts per 1,000 scored decisions | `1000 * FP / N`; operational burden on the labeled cohort |
| False-positive rate | `FP / (FP + TN)`; distinct from the false-alert share `FP / (TP + FP)` |
| PR curve and average precision | Describe recall/precision tradeoffs under class imbalance; show cohort prevalence as context |
| Confusion counts and alert rate | TP, FP, FN, TN, total scored N, and alerts/N; make denominators visible |
| Probability quality | Brier score and reliability plot; calibration is evaluated on the selected labeled cohort |

Report precision, false alerts and alert rate at validation recall targets of **80%, 90% and 95%** as comparison points, not approved minimum performance requirements. For each attainable target, choose the highest alert threshold meeting it; state when it is unattainable. Carry that frozen threshold into test evaluation without readjustment. Minimum recall and acceptable false-alert burden remain open decisions. Do not declare a deployment pass based on accuracy, ROC-AUC, or an invented numeric requirement.

If a subgroup has no positives, recall is undefined; if there are no alerts, precision is undefined. Report denominator counts and NA rather than zero or an apparent perfect score. Distinguish missing labels from predicted negatives. Primary scores should use the natural cohort frequencies; any training resampling must not be applied to validation/test scoring distributions.

## Continuous peak evaluation

Report MAE in customers, RMSE in customers to expose large errors, and mean signed error (`prediction - observation`) to reveal systematic underprediction. Also report MAE and signed error on observed peaks >5,000 and >10,000. Include observed-versus-predicted plots and absolute-error quantiles. If a transformed target is explored, evaluate back in customer units and document inversion; do not silently cap high outcomes. Avoid MAPE as a primary metric because observed zeros and small counts make it unsuitable for this target. No transformation or regression algorithm is prescribed here.

Use a simple past-outage persistence comparison only under the same availability assumptions as the candidate model. Score comparisons on the same labeled origins, report any additional loss from missing predictors, and do not advantage one model by dropping its hard-to-predict rows silently.

## Subgroups, dependence and preparation usefulness

Report both horizons separately, with state, calendar period, county-size and county-level counts, retention, recall, precision and severe-case errors. Housing quartiles are retrospective evaluation strata, not forecast inputs. Include overall window-weighted scores and a county-macro summary over counties with defined metrics; show how many counties were excluded from the macro summary.

Overlapping windows repeat outage episodes, and one storm can affect multiple counties. Do not attach independent-row confidence intervals. If uncertainty is estimated, use contiguous time-block resampling across all counties jointly, report the chosen block length and sensitivity to longer blocks; a proposed starting comparison is 7 versus 14 days, not an established independence guarantee.

For a secondary preparation view, group consecutive severe-labeled county origins into runs, broken by unknown/ineligible gaps, and report the proportion of runs receiving an alert. Count these as **observed severe runs**, not verified independent incidents. For an onset/lead-time analysis, require observed preceding nonsevere conditions; otherwise label onset unknown. Report repeated alerts and missed runs separately from window recall. This analysis is a teammate evaluation requirement, not an EDA claim that incident-level detection has been measured.

## What remains before deployment

- Freeze target and cohort definitions, splits, as-of feature assumptions and scoring denominators before modeling. No result in the old storm-conditioned notebook supersedes these forecast definitions.
- Require source hashes and the geography overlay version in reproducible runs. The overlay affects retrospective storm attribution, not outage labels; unmatched storms never remove county-hours from the forecast population.
- Existing inputs do not establish operational outage latency or historical feature release times. Use [the availability matrix](feature_availability.md); weather-event narratives and realized storm attributes are excluded from the advance forecast feature set.
- Coverage selection, housing-proxy uncertainty, the large outage tail and prior test exposure remain material limitations.
- Before any deployment claim, resolve measured input latency, acceptable false-alert burden and a minimum recall requirement, and validate outside the selected-cohort historical setting. An EDA handoff can proceed with those limitations documented; operational readiness cannot.
