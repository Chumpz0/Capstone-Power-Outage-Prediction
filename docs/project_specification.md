# Project specification: Southeast major-outage prediction

Status as of 2026-10-23, after checkpoints A and B. Decisions are recorded with dates and evidence in [decision_log.md](decision_log.md). This page lists what is approved and what is still open. The repository ends at completed EDA. Previous modeling code and results were removed at the user's request; the teammate owns baseline modeling.

## Purpose

Predict the probability that a storm causes a **major power outage** in a county in Alabama, Florida, Georgia, North Carolina, South Carolina or Tennessee. As a supporting task, estimate the **peak number of customers out at the same time**. Outages come from EAGLE-I county observations. Weather comes from NOAA Storm Events and, if the before-arrival design is chosen, from archived NWS alerts.

Previous national and Southeast model experiments are retired. Their removal does not undo prior use of 2024 for development or the disclosed earlier 2025 exploration.

## Approved decisions

| Item | Decision |
|---|---|
| Primary objective | Probability of a major outage in a county outcome window |
| Supporting objective | Peak simultaneous customers out (EAGLE-I's maximum quarter-hour count, not unique customers) |
| Geography | AL, FL, GA, NC, SC, TN (534 counties) |
| Prediction geography | County. Zone-based weather enters through historical NWS zone-county mappings (stage 09). |
| Outcome window | Fixed 24 hours at first; 72 hours for alert rows since decision D-21 |
| Severity measure | Customer counts are primary. Customer totals are kept where available and validated before any percentage is computed. |
| Evaluation | Train on years through 2023, validate on 2024, final evaluation once on 2025. Chronological splits; overlapping storms stay inside one split. |
| 2025 | Previous exploration of 2025 is disclosed. No further target-driven work on 2025 until the final evaluation. |
| Tracking and sharing | Deferred to the teammate; no experiment tracking is required for the EDA handoff. |
| Causation | Results describe association with storms; pre-existing outages are included and reported separately. |

## Approved decisions after the checkpoints (2026-10-23)

| Item | Decision |
|---|---|
| Audience and action | Utility operations and mutual-aid planners deciding where to pre-position crews 12-48 hours ahead; missed major outages are the costly error |
| Prediction time | First storm-family NWS watch or warning issued for the county (alert-triggered rows, primary). NOAA-recorded onsets are a retrospective comparison. |
| Outcome window | `[t0, t0 + 72 h)` for alert rows, with non-overlapping 72-hour county windows; `[t0, t0 + 24 h)` for onset rows |
| Major outage | At least 5,000 customers out at the same time in the outcome window |
| Non-major | `cov90`: at least 90% of scheduled runs observed (260 of 288 for alert rows) and no gap over 1 hour; otherwise unknown and excluded |
| Outages underway at `t0` | Excluded from the primary target and reported separately |
| Evaluation population | Covered windows (coverage holds whatever the label); models are fit on the same population |
| Alert threshold | Highest probability with validation recall at least 0.80 |
| Storm scope | Storms only; heat and cold excluded |

## Still open

| Decision | What it changes | When |
|---|---|---|
| Severity population | Provisional: covered windows | before freezing the protocol |
| Implausible-peak counties | Kept and reported as a subgroup for now | before freezing the protocol |
| Baseline and evaluation design (O-12) | Teammate to declare baseline, metrics, calibration approach and success criteria | before new experiments |
| Step 9 prerequisites | Tracking and sharing accounts, repository, redistribution terms | before step 9 |

## Rules for prediction rows and information

- **Row:** one county prediction opportunity at a prediction time `t0`. Overlapping weather records in the same county are merged so that each county outcome window is counted once (stage 10).
- **Outcome window:** `[t0, t0 + 72 h)` for alert rows (288 scheduled EAGLE-I quarter-hour runs) and `[t0, t0 + 24 h)` for onset rows (96 runs).
- **Labels:** *major* when an observed value crosses the threshold, *non-major* only when the coverage rule holds, otherwise *unknown*. Absent EAGLE-I rows are never read as zero outages. EAGLE-I does not distinguish zero from a collection gap.
- **Information at `t0`:** only sources whose timestamp is at or before `t0`. EAGLE-I runs that started at or before `t0 - 15 min`, alert products issued by `t0`, history computed from earlier data only. Realized duration, damage, narratives, later storm records and final alert status are excluded. Columns realized after `t0` carry a `retro_` prefix.
- **Retrospective NOAA records cannot establish operational availability.** If the onset design is kept, it is labeled a retrospective prototype.

## Checkpoints and sequence

Current endpoint: data preparation, candidate features and training-only EDA are complete. Modeling steps below are future teammate work.

1. **Checkpoint A** (after step 2): agree on audience, action and prediction time before building final features.
2. **Checkpoint B** (after steps 3-4): approve the major threshold, coverage rule and pre-existing-outage treatment before comparing models. The threshold is chosen from label feasibility and the intended decision, not from model scores.
3. Steps 6-8 (features, benchmarks, controlled experiments) use only training and validation data.
4. Freeze the model, thresholds, features and evaluation protocol, then score 2025 once (step 9).
5. Completeness review with a pass / partial / missing checklist (step 10).

## Out of scope

- Unique affected customers, restoration time, or damage estimates.
- Causal attribution of outages to storms.
- Operational deployment. Any operational claim needs inputs that were actually available at prediction time.
