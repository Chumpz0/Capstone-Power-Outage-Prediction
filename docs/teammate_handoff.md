# Teammate baseline handoff

This contribution provides completed EDA and reproducible inputs; it fits no baseline. The teammate owns model choice, preprocessing, calibration, development folds and declared experiments. Old local model grids/results were retired, not adopted as requirements.

## Start here

1. Read the [EDA report](../reports/eda/major_outage_eda.html), [dataset card](dataset_card.md) and [feature dictionary](feature_dictionary.md).
2. Install the tested EDA environment from `requirements-eda.txt`. Add modeling dependencies from the team's `requirements.txt` as needed.
3. Run `scripts/download_eda_data.py` after local `hf auth login`, once the release receipt exists. It installs six Parquet files and the manifest under `data/04_model_input/southeast_v1/`, validating exact hashes/revision. Alternatively use [pipeline_setup.md](pipeline_setup.md).
4. Use `src.major_outage.load_major_outage_xy('train')` and `('validation')`. Returns are `(X, y, groups)`; predictors are allowlisted and rows require `covered_window`. The audit loader can expose all development windows, but those are not the approved model population.
5. Declare experiments in [evaluation_protocol.md](evaluation_protocol.md) before fitting. Freeze all choices before one final evaluation on 2025.

## Contract and limitations

- Five-character county FIPS; UTC prediction timestamps; non-overlapping county windows.
- Alert rows: 72 hours, major >=5,000; `cov90` requires 260/288 observations and no gap >1 hour. Ongoing major outages excluded.
- Preprocessing fits training only. Chronological folds must purge overlapping boundary windows; resample dates or storm episodes for uncertainty.
- Coverage, peaks, labels, `retro_*` and subgroup/audit fields must never enter X. `feature_columns()` validates the allowlist against the manifest.
- Missing rows do not mean zero. Observed partial-window peaks are lower bounds, not exact maxima.
- O-7 severity population/completeness and O-8 implausible-peak counties remain open; baseline evaluation design is O-12.
- Prior 2024 development and earlier 2025 exploration remain disclosed. No 2025 outcomes may be explored during baseline development.
- Legacy EDA used estimated denominators, 5% thresholds, event-duration windows and zero-filled missing aggregates. Its outputs are cleared and execution is disabled under `notebooks/legacy/`.

## Small release

Six input Parquets (train/validation with outcomes and test features only, for alert and onset designs), the stage-13 manifest, attribution/dataset README, and SHA-256 inventory. No raw observations, underlying 2025 outcome tables, models or credentials are uploaded.

The remote prefix `southeast_v1/releases/2026-10-23-eda-v2` is immutable. The publisher refuses to overwrite it and requires a private destination. Existing legacy paths and the root dataset card remain intact. After publication `docs/eda_release.json` pins the repository, commit and checksum-manifest hash. Verify upload by downloading into a clean directory and comparing hashes.

The small release suffices for current EDA and baseline development. Full reconstruction needs preserved raw archives and reference downloads; exact NOAA revision filenames can disappear upstream. Integration does not claim a fresh full raw rebuild. Shared configuration wiring changes may invalidate preparation caches without changing approved label semantics; run stage checks before relying on caches.

Legacy `data/processed/` and current `data/04_model_input/southeast_v1/` are separate schemas. Existing `src.data` loaders continue to serve legacy files and must not silently substitute for approved inputs. `src/config.py` imports authoritative paths/geography/splits from the pipeline.
