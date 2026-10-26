# Success Criteria: P(county outage | storm)

Agreed 2026-10-25, **before any model was trained**. Models are judged on validation (2024). The winning model is then checked **once** on test (2025). These numbers don't change after we see results.

Unit and target: county × storm episode, positive = ≥5% of customers out within 24 h (see [04 summary](04_modeling_table_summary.md)). Intended use: **utility crew staging**, i.e. pre-positioning crews in the counties most at risk.

---

## The bar: naive rules on 2024 validation (no ML)

| Naive rule | PR-AUC | ROC-AUC | Brier | Recall @ top 10% |
|---|---|---|---|---|
| Always predict the training rate (14.1%) | 0.165 | 0.50 | 0.138 | 9.3% |
| Storm family rate | 0.312 | 0.72 | 0.125 | 23.2% |
| County history rate | 0.368 | 0.75 | 0.124 | 26.9% |
| **Storm family rate × county history** (reference) | **0.469** | **0.80** | **0.112** | **32.6%** |

Validation positive rate is 16.5%. The Brier score of always predicting that rate is 0.138.

How the rules are computed (code: `naive_rules()` in [src/evaluation.py](../src/evaluation.py), run in notebook 05). All rates come from training rows only.
- **Storm family rate:** the highest training outage rate among the storm families present in the episode.
- **County history rate:** `hist_outage_rate`.
- **Reference:** family rate × county history ÷ training rate, clipped to [0, 1].

> **Correction (2026-10-25, before any model was trained):** the reference row first said PR-AUC 0.472 and ROC-AUC 0.81. That came from an ad-hoc calculation that wasn't saved as code and can't be reproduced exactly. The reproducible rule above gives 0.469 and 0.80, with the same Brier, Brier skill and recall. The targets below are unchanged.

---

## A model is "good" if it meets all five

| # | Criterion | Target | Reference rule |
|---|---|---|---|
| 1 | **Ranking:** PR-AUC on validation | **≥ 0.52**, and the 95% bootstrap CI of the gain over the reference rule (resampled by storm episode) stays above 0 | 0.469 |
| 2 | **Calibration:** Brier skill score vs always predicting the base rate | **≥ 0.25** | 0.19 |
| 2b | Mean prediction vs observed rate | within **±2 points** overall, and within **±5 points** in each predicted-probability decile | n/a |
| 3 | **No hidden failures** | PR-AUC above the storm-family-rate rule **within each major storm family** (thunderstorm wind, winter, high wind, hail, flood, tornado) **and each FEMA region** | n/a |
| 4 | **Holds up on new data** | test (2025) PR-AUC within **0.05** of validation | n/a |
| 5 | **Crew staging:** share of real outages caught when flagging the top 10% of county-episodes | **≥ 40%** (the most possible is ~60%) | 32.6% |

The criterion 5 target was proposed by Claude after the use case was chosen. It is open to change before step 05 results come in.

**Reported for every model, not pass/fail:** ROC-AUC, the calibration curve, precision at top 10%, and results by year and by storm family.
