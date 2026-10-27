# 06 – EDA of the ML-Ready Dataset

Notebook: [notebooks/06_ml_dataset_eda.ipynb](../notebooks/06_ml_dataset_eda.ipynb) · Tables: [reports/ml_eda_tables/](ml_eda_tables/) · Figures: [reports/figures/ml_eda/](figures/ml_eda/) · Run 2026-10-25

**Outputs:** EDA tables and figures linked above, plus the ML-ready dataset in Hugging Face `DSCI-Capstone/county-outage-data`, `processed/model_table/`.

---

## [GOAL]

Profile the modeling table from step 04 the way the models will see it: one row per county × storm episode, 36 features, target = ≥5% of customers out within 24 h. Answer:
- what the dataset holds, and how it was built (lineage, operations, metadata)
- how each feature is distributed, and how it relates to the target
- which features are redundant, missing in meaningful ways, or unstable across the train / validation / test years

**Rules (unchanged from earlier steps):**
- Feature–target relationships use training rows only (2017–2023).
- Splits are compared on features only.
- The 2025 labels are never read.

---

## [FINDINGS]

The evidence is in the [EDA notebook](../notebooks/06_ml_dataset_eda.ipynb), [result tables](ml_eda_tables/), and [figures](figures/ml_eda/).

The [consolidated EDA report](eda_report.md) organizes these existing results and the earlier event-level EDA into a credited report template.

1. **Size:** 271,644 rows (train 208,482 · validation 31,064 · test 32,098), 3,073 counties and 54,580 episodes in train. The positive rate is 14.1% in train, **rising from 11% (2017) to 17% (2023)**, and 16.5% in validation.
2. **Strongest single signals (training ROC-AUC alone):**
   - county past outage rate 0.74, past mean peak 0.73
   - worst share out in the previous 24 h 0.67
   - reports in the whole episode 0.65, thunderstorm wind reports 0.64
   - No single feature reaches the reference rule's 0.80.
3. **Storm family:** tropical 62.5%, ice storm 33%, tornado 29%, thunderstorm wind 25%; winter, cold, wildfire and other 3–8%.
4. **Location coding:** county-coded storms 17.9% positive vs zone-coded 9.9% (partial-county zones 7.8%).
5. **Already-out counties:** ≥5% out the hour before → 89% positive (1.2% of rows).
6. **Missing values are informative.** No wind measurement → 8.6% positive vs 22.1% with one. `max_hail_in` is 87% missing, `max_hurricane_cat` 99.9%.
7. **Non-linear effects:**
   - County size has a hump (~18% for mid-sized counties, ~5% for the largest).
   - Short, intense county episodes are riskiest (22% under an hour vs 7% for multi-day episodes).
   - Risk follows the hour of day (10% at midnight → 16% in the afternoon and evening).
8. **Near-duplicate pairs** (|ρ| ≥ 0.93): `episode_counties`/`log_episode_counties`, `log_customers`/`log_pop`, `n_zone_coded`/`share_zone_coded`, `hist_outage_rate`/`hist_mean_peak`.
9. **Drift:**
   - Every feature has PSI < 0.1 between train and validation/test, **except `dqi` (PSI 3.7)**. Its values only exist for 2018–2022, so it works as a year marker.
   - `hist_episodes` is ~17% larger outside train, from the leave-one-year-out design (PSI < 0.06).
   - 2025 has only 9 tropical rows.
10. **Checks passed** ([checks.csv](ml_eda_tables/checks.csv)): unique keys across splits, no missing target, target = (`retro_peak_24h` ≥ 5%), no `retro_` feature, no target in the test file, split years don't overlap, and the dictionary covers every column.

**Provisional engineering choices:**
- Up to 10 quantile bins per feature for the rate curves.
- PSI computed with training deciles and thresholds of 0.1 / 0.25.
- `max_hurricane_cat` left out of the drift chart because it's 99.9% empty.
