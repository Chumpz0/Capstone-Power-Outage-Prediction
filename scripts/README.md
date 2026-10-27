# Python scripts

All six notebooks have standalone Python counterparts in this folder. These are readable copies of the existing analysis code, not wrappers that execute notebooks. They retain the original calculations and section comments, with source-cell references for comparison. The notebooks remain available with their saved outputs and explanatory text.

## Available scripts

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| [01_data_audit.py](01_data_audit.py) | Audit raw outage, storm, and reference data | Raw data and reference files; reuses `work/audit.duckdb` if present | `reports/audit_tables/`, printed checks, three charts in `reports/figures/audit/` |
| [02_data_prep.py](02_data_prep.py) | Clean outages/storms, harmonize geography, construct observation flags | Raw data and reference files; reuses or builds the DuckDB cache | Cleaned tables under `processed/`, checks under `reports/prep_tables/` |
| [03_eda.py](03_eda.py) | Explore cleaned training-year outage and storm data | Prepared tables from step 02 | Eight charts under `reports/figures/`, printed statistics |
| [04_modeling_table.py](04_modeling_table.py) | Build county-by-episode features and labels | Prepared tables from step 02 | `processed/model_table/`, `reports/model_table_tables/` |
| [05_baseline_models.py](05_baseline_models.py) | Evaluate naive rules and the existing four-feature logistic regression | Training and validation modeling tables; `src/evaluation.py` | Three CSVs under `reports/baseline_tables/`, printed metrics and threshold comparisons |
| [06_ml_dataset_eda.py](06_ml_dataset_eda.py) | Profile modeling features, distributions, missingness, and drift | Training, validation, and test-feature tables | `reports/ml_eda_tables/`, eight charts under `reports/figures/ml_eda/` |

## Running

Use the project's Python environment and [requirements](../requirements.txt). From the repository root, with the virtual environment activated:

```bash
python scripts/01_data_audit.py
python scripts/02_data_prep.py
python scripts/03_eda.py
python scripts/04_modeling_table.py
python scripts/05_baseline_models.py
python scripts/06_ml_dataset_eda.py
```

On Windows without activation, replace `python` with `.venv\Scripts\python.exe`. Each script also accepts `--help` without loading data. Paths resolve relative to the script's repository location, so scripts can be launched from another working directory using an absolute script path.

The dependency order is **02 -> 03/04 -> 05/06**. Step 01 is the raw-data audit; step 02 can create its own cache. Step 03 documents EDA but does not generate inputs required by 04. Step 06 can run before 05. If the prepared or modeling tables already exist locally or on Hugging Face, earlier steps do not have to be rerun.

The local-file-first/Hugging Face fallback behavior is preserved. Accessing the private dataset requires `hf auth login` and the appropriate dataset permission. The preparation stages process large datasets and write to the same output locations as their notebooks. Step 02 replaces the generated `outages_15min` and `outages_hourly` partitions; step 04 rewrites modeling tables. These commands are not a read-only report preview.

Step 01 preserves the original audit's raw-data scope, including 2025 observations. Step 04 constructs the separate test-label file but does not report its label rate. Step 05 reads training and validation only; step 06 reads test features only. Neither 05 nor 06 opens the test-label file.

## Changes for standalone execution

- Each file has a `main()` entry point and a main guard, so importing it does not load data or run the pipeline.
- Notebook-only implicit result displays become `print(...)`; the one explicit `display(...)` call also becomes terminal output.
- Matplotlib uses the non-interactive `Agg` backend. Figures are saved and closed rather than shown in a notebook or GUI. The audit's previously inline-only figures are saved under `reports/figures/audit/`.
- Project paths use `__file__` rather than the current working directory.
- Step 02 checks the resolved directory before replacing either generated outage partition folder.
- Step 05 keeps W&B off by default. To explicitly upload metrics, summary tables, and charts, run `python scripts/05_baseline_models.py --log-to-wandb` after configuring W&B access.

Notebook markdown and saved cell outputs are not copied into the scripts as historical results. Read the [reports](../reports/), [EDA report](../reports/eda_report.md), and [data card](../reports/dataset_card.md) for interpretation. No notebook runner or Jupyter kernel is required to execute these files.

## Keeping the two versions aligned

These are separate source copies, not automatically synchronized exports. If a calculation changes, update its script and notebook counterpart together. Shared model-evaluation logic remains in [src/evaluation.py](../src/evaluation.py). Source-cell comments refer to the notebook cell positions when these scripts were created.

## Validation

All six scripts were checked for Python syntax, safe import, and `--help` execution from another working directory. Steps 05 and 06 were run in an isolated workspace with copies of the existing training, validation, and test-feature files; their CSV outputs were compared with the saved notebook results, and all eight step-06 figures were produced. W&B uploads were disabled and no test-label file was included.

The full raw-data audit, preparation, event-level EDA, and modeling-table rebuild (01-04) were not rerun as part of this conversion. Their standalone versions preserve the notebook logic, but full-data runtime verification remains separate from syntax and entry-point checks.
