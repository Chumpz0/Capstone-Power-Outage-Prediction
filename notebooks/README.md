# Notebooks

| Notebook | What it covers |
| --- | --- |
| `01_quality_outages.ipynb` | EAGLE-I outage data quality: nulls, missing rows, coverage, customer counts |
| `02_quality_storms.ipynb` | NOAA Storm Events data quality: nulls, event types, magnitudes, zone-to-county mapping |
| `03_eda_storm_outage.ipynb` | Retrospective storm/outage associations using audited mappings; descriptive percentage targets |
| `04_eda_model_readiness.ipynb` | Checkpoints 1-3: training-period observation reliability and targets; geography integrity and forecasting criteria |

Name new notebooks with a number prefix so they sort in order, e.g. `05_features.ipynb`.

## Running them

1. Set up the repo's environment once (see the main README): `python -m venv .venv`, then
   `pip install -r requirements.txt`.
2. **Pick this repo's `.venv` as the kernel.** In VS Code: *Select Kernel* -> *Python Environments* ->
   `.venv` under `Capstone-Power-Outage-Prediction`. A kernel from another project may be missing packages
   (e.g. `jinja2`, which pandas needs for the formatted `.style` tables).
3. Before committing, use *Restart* + *Run All* so the saved outputs come from one clean run.

## Loading data

Use the loaders in `src/data.py`; they read `data/processed/` if you have it and stream from Hugging Face otherwise:

```python
import sys
sys.path.insert(0, "..")
from src.data import load_outages, load_storm_counties, load_census

out = load_outages([2024], columns=["fips_code", "customers_out", "run_start_time"])
```

For DuckDB queries, `duckdb_path("eaglei/outages_*.parquet")` gives a path (or glob) DuckDB can read.
Shared chart colors and helpers are in `src/plotting.py`.

## Current EDA conventions

The storm/outage join and grouped burden calculations live in `src/eda.py`;
`scripts/audit_data_review.py` uses the same functions without executing notebook cells.
Unknown event windows retain null outcomes. Rates exclude those unknown windows,
but still include partially observed windows, so they describe observed severity.
The preceding 24-hour mean uses available hours; its observation count is retained.
Burden counts each county-hour once within a group; different groups can overlap.
NOAA episodes stay separate even when their narrative labels name the same storm.
The housing multiplier and event horizon remain provisional EDA assumptions.

Focused calculation checks: `python -m unittest discover -s tests -v`.
The data review in `docs/data_review.md` is the pre-cleanup baseline; refreshed
notebook outputs and `docs/data_audit.json` reflect these corrected conventions.

## How to reproduce the EDA

Follow the commands in the [teammate handoff](../docs/teammate_handoff.md).
They build the local summaries and corrected storm mapping, run the tests,
execute notebooks 01-04, and refresh metadata.

For interactive runs, build those local outputs first and select this repo's
`.venv` kernel. Notebook 04 can build missing summaries. Set `REBUILD = True`
to regenerate them after source data or calculation code changes; older summaries
without code fingerprints also need rebuilding.

## Where to find the results

- [EDA report](../docs/eda_report.md): findings and next steps.
- [Modeling criteria](../docs/modeling_criteria.md): targets, splits and evaluation.
- [Feature availability](../docs/feature_availability.md): what can be used at prediction time.
- [Metadata guide](../docs/eda_metadata.md): field definitions and calculation rules.

Detailed summaries are generated under the ignored `data/processed/eda/` folder.
The scripts also generate `docs/checkpoint_1_observation_reliability.json`,
`docs/checkpoint_2_customer_targets.json` and `docs/checkpoint_3_geography_audit.json`.
These contain aggregate results and source/code fingerprints.

The geography output preserves original IDs and mapping methods. Notebook 02
shows the mapping problems; notebook 03 uses the corrected event table.
Notebook 03's percentage-based storm comparisons are descriptive. Notebook 04
provides the future 6/12-hour customer-count targets used for modeling.

Charts remain embedded in the notebooks. The finalizer records execution
results in `docs/eda_validation.json` without exporting separate image files.
