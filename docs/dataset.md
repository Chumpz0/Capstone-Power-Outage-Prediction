# Dataset and modeling definitions

This capstone studies substantial county outages coinciding with recorded storms.
It is a retrospective research dataset, with utility crew staging as motivation;
it does not establish operational forecasting or storm causation.

## Release and access

Sources: [EAGLE-I](https://figshare.com/articles/dataset/24237376),
[NOAA Storm Events](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/),
Census Vintage 2023 county population, EAGLE-I MCC/DQI/coverage references,
dated NWS zone crosswalks, and Connecticut population weights.

The private dataset is [DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data).
Access requires organization permission and `hf auth login`. Remote reads default
to revision `5c01325258aa922bf395da1e39acda7c234fefcf`; there is no silent latest fallback.
Local files take precedence and are hashed for model provenance. A local hash
does not establish that a file matches that remote revision.

On 2026-11-16, authenticated pandas/DuckDB reads verified the pinned raw,
reference and prepared paths. The current local train, validation and test-feature
files also matched the remote LFS SHA-256 hashes exactly. This verifies that
snapshot; future local files must still be identified independently.

| Split | Years | Rows | Outcomes |
|---|---|---:|---|
| Train | 2017–2023 | 208,482 | Available |
| Validation | 2024 | 31,064 | Available for development |
| Test | 2025 | 32,098 | Final evaluation completed; excluded from ordinary workflows |

The unit is county FIPS × NOAA episode. Train/validation have six identifiers,
36 candidate features, the target, and three `retro_` fields (46 columns). Test
features contain identifiers/features only (42 columns). The selected boosting
candidate uses nineteen of the candidate features. Field definitions and training
statistics are returned by `analysis.feature_dictionary()` in the EDA notebook.

The frozen retrospective predictor was evaluated on all 2025 rows on 2026-11-25.
Base-model fitting uses 2017-2023; the selected sigmoid calibrator uses only
January-June 2024. The dataset and labels were not rebuilt for final evaluation.
See [final results](evaluation.md#final-2025-results-2026-11-25). Ordinary analysis
and training continue to exclude test outcomes.

## Preparation and labels

Raw inputs are immutable. Preparation parses times, snaps outage observations
to a 15-minute grid, deduplicates by maximum reading, maps Connecticut planning
regions to legacy counties, joins customer denominators, caps implausibly large
outage counts, and aggregates hourly maxima. MCC is the default denominator;
missing or implausible values use population × the median customers/person ratio.
Provisional county ratio bounds are 0.1–5; state ratios below half the national
median also trigger fallback. Original counts and capped-reading flags are retained.

NOAA local standard times become UTC. County-coded reports map directly;
zone-coded reports use the crosswalk version in effect at the start. The oldest
crosswalk covers earlier dates. Modeling includes lower 48 + DC and excludes
drought, heat and excessive heat triggers.

For first county report `t0`, the target is peak outage share ≥ 5% in
`[floor_hour(t0), floor_hour(t0) + 24 hours)`, using hourly maxima. The first
hour can include outages before exact `t0`. Pre-storm inputs use hours −24…−1;
`pct_out_at_start` is the preceding-hour share. Already-out counties remain eligible.

A county-day is considered observed if the county appears somewhere that year,
national snapshots cover at least 90% of the day, and the county-day is not
all-null. Eligibility requires observation on the dates containing label hours
0 and 23. Missing hourly readings then mean zero under this coverage heuristic,
not proven continuous monitoring. Pre-storm features are missing if the preceding
day was unobserved. Quality references cover 2018–2022 and use nearest-year mapping.

## Feature availability and limitations

Whole-episode storm counts, intensity and extent are retrospective proxies and
can include later reports. `retro_` fields are analysis-only, but excluding them
does not establish forecast-time availability of other features.

County history uses training labels only. Validation/test use all training years;
training county sums exclude the row's entire year. Smoothing uses 20 pseudo-episodes
and full-training priors. Thus a row's label can influence its history through the
prior, and other training years may be later than its date. This is not a strictly
historical pipeline. This refactor preserves that definition; chronological history
and exact-minute labels require separately rebuilt and revalidated datasets.

Missing storm intensity often reflects event type; preserve missingness until
training-only preprocessing. Population gaps reflect legacy Connecticut geography.
Local hour approximates the state's majority standard-time zone. Measurement
coverage, estimated denominators, correlated county rows, and changing storm mix
limit generalization. Rebuild histories within folds if changing the split design.

## Storage and provenance

Local raw/reference/interim/modeling/holdout data live under `data/`; model runs
live under `models/`; caches under `.cache/`. The resolver also recognizes legacy
local paths and maps new paths to the unchanged remote layout. Preparation uses
content-identified raw caches. New outputs are validated in staging before publication,
with the previous generation retained. Creating holdout labels requires the explicit
feature-build `--write-holdout` option; normal notebooks never read them.

Model-table rows are ordered by `t0`, `fips`, then `episode_id`. The final key
breaks ties between episodes sharing the same county/start time, so rebuild
ordering does not depend on SQL aggregation order. Released files were not
replaced when this ordering fix was introduced.

Upstream release dates, reference-file provenance, redistribution terms, and the
intended final release name still require team completion. Author/contact details
and the extended release description are retained in the [dataset card](../reports/dataset_card.md).
The original dataset card was adapted from the
[PAIR Data Cards Playbook](https://github.com/PAIR-code/datacardsplaybook).
See [evaluation policy](evaluation.md) and [EDA](../notebooks/02_eda.ipynb)
for supporting context.
