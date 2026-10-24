# Notebooks

| Notebook | What it covers |
| --- | --- |
| `01_quality_outages.ipynb` | EAGLE-I outage data quality: nulls, missing rows, coverage, customer counts |
| `02_quality_storms.ipynb` | NOAA Storm Events data quality: nulls, event types, magnitudes, zone-to-county mapping |
| `03_eda_storm_outage.ipynb` | Storm-outage join in DuckDB: which storms cause outages, when, where, and how big |

Name new notebooks with a number prefix so they sort in order, e.g. `04_features.ipynb`.

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
