# Evaluation safeguards for the baseline handoff

Status: EDA endpoint, 2026-10-23. Previous modeling implementations, fitted artifacts and results have been removed at the user's request (D-25). The teammate owns the baseline. No model family, hyperparameter grid, calibration method or success criterion is selected by this document.

## Approved task and population

- Primary: alert-triggered county windows, 72 hours, in AL, FL, GA, NC, SC and TN.
- Major: at least 5,000 simultaneous customers out. Non-major needs `cov90`: at least 260 of 288 runs and no gap over one hour. Missing EAGLE-I rows are never zeros.
- Fit and evaluate on `covered_window`: coverage holds and the outage was not already major at `t0`. This flag selects rows; it must never be a predictor.
- NOAA-onset rows are a retrospective comparison with 24-hour windows (87 of 96 runs for `cov90`).
- Supporting severity target: observed peak customers. Population and completeness policy remain open (O-7); a partially observed peak is not an exact maximum.
- Results describe the covered population, which is selected toward larger counties and recent years, not all storms.

## Evaluation boundaries

- Train through 2023, validate on 2024, score the held-out 2025 test once after freezing the model, features, thresholds and protocol.
- 2024 was previously used for model development and informed retained design decisions. Removing those experiments does not restore an untouched validation set.
- Earlier 2025 exploration remains disclosed. No further target-driven exploration, outcome-window summaries, label counts or tuning on 2025 before the frozen final evaluation. Structural coverage audits are allowed.
- Test files contain features only. Do not build test labels during baseline development.
- Fit preprocessing on training data only. Use chronological development splits and keep overlapping storm observations together; purge windows crossing fold boundaries.
- Estimate uncertainty by resampling storm episodes or dates, never independent event rows.
- Use only features available at `t0`; exclude outcome, `retro_*`, final alert state and analysis-only fields. Candidate features are listed in the dataset manifest and feature dictionary.
- Approved operational threshold rule: highest validation probability threshold achieving recall at least 0.80 (D-23). No numeric probability threshold has been selected for the teammate's baseline.

## Teammate decisions before experiments

Declare the baseline, predictors, preprocessing, development folds, metrics, selection rule, calibration approach (if any), uncertainty procedure and success criteria here before running experiments. The previous model grids and provisional C1/C2/C3/S1 criteria are retired, not requirements for the new baseline (O-12).

Resolve severity population/completeness (O-7) and implausible-peak county treatment (O-8) before freezing the final protocol. Changes to approved label or population rules require an explicit decision; do not change thresholds to improve scores.

## Experiment declarations

None active. Baseline implementation is assigned to the teammate.
