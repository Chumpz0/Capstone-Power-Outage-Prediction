# Success Criteria: P(county outage | storm)

Agreed 2026-10-25, **before any model was trained**. Models are judged on validation (2024). The winning model is then checked **once** on test (2025). These numbers don't change after we see results.

Unit and target: county × storm episode, positive = ≥5% of customers out within 24 h (see [04 summary](04_modeling_table_summary.md)). Intended use: **utility crew staging**, i.e. pre-positioning crews in the counties most at risk.

---

## The bar: naive rules on 2024 validation (no ML)

| Naive rule | PR-AUC | ROC-AUC | Brier | Recall @ top 10% |
|---|---|---|---|---|
| Always predict the training rate (14.1%) | 0.165 | 0.50 | 0.138 | n/a |
| Storm family rate | 0.312 | 0.72 | 0.125 | n/a |
| County history rate | 0.368 | 0.75 | 0.124 | n/a |
| **Storm family rate × county history** (reference) | **0.472** | **0.81** | **0.112** | **32.6%** |

Validation positive rate is 16.5%. The Brier score of always predicting that rate is 0.138.

---

## A model is "good" if it meets all five

| # | Criterion | Target | Reference rule |
|---|---|---|---|
| 1 | **Ranking:** PR-AUC on validation | **≥ 0.52**, and the 95% bootstrap CI of the gain over the reference rule (resampled by storm episode) stays above 0 | 0.472 |
| 2 | **Calibration:** Brier skill score vs always predicting the base rate | **≥ 0.25** | 0.19 |
| 2b | Mean prediction vs observed rate | within **±2 points** overall, and within **±5 points** in each predicted-probability decile | n/a |
| 3 | **No hidden failures** | PR-AUC above the storm-family-rate rule **within each major storm family** (thunderstorm wind, winter, high wind, hail, flood, tornado) **and each FEMA region** | n/a |
| 4 | **Holds up on new data** | test (2025) PR-AUC within **0.05** of validation | n/a |
| 5 | **Crew staging:** share of real outages caught when flagging the top 10% of county-episodes | **≥ 40%** (the most possible is ~60%) | 32.6% |

The criterion 5 target was proposed by Claude after the use case was chosen. It is open to change before step 05 results come in.

**Reported for every model, not pass/fail:** ROC-AUC, the calibration curve, precision at top 10%, and results by year and by storm family.
