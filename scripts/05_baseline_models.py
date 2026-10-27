"""Standalone Python version of notebooks/05_baseline_models.ipynb.

Run with the project virtual environment. See scripts/README.md for inputs,
outputs, execution order, and differences from the notebook.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-to-wandb", action="store_true", help="Upload evaluation metrics, summary tables, and charts to W&B.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # Save figures without requiring a notebook or GUI.


    # Source: 05_baseline_models.ipynb, code cell 2
    # =============================================================================
    # 05 - BASELINE MODELS: P(county >= 5% out within 24 h | storm episode)
    # Results are saved in reports/baseline_tables/.
    # Judged on 2024 validation against reports/success_criteria.md (test 2025 is not touched here).
    # Optional W&B logging sends metrics, small summary tables, and charts to "CAPSTONE MODEL".
    # No data or labels are uploaded to W&B. Shared checks live in src/evaluation.py.
    # Inputs: processed/model_table/ (from notebook 04). Outputs: reports/baseline_tables/
    # =============================================================================

    # Source: 05_baseline_models.ipynb, code cell 3
    # ---- 0. Setup ---------------------------------------------------------------
    import sys
    from pathlib import Path

    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.evaluation import evaluate, log_run, naive_rules  # noqa: E402

    TABLES = ROOT / "reports" / "baseline_tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    LOG_TO_WANDB = args.log_to_wandb  # Off by default; enable explicitly with --log-to-wandb.

    # Inputs: the local file when it is in the repo, otherwise streamed from the Hugging Face dataset
    HF_DATASET = "hf://datasets/DSCI-Capstone/county-outage-data"


    def src(rel):
        """Local path if the file is in the repo, otherwise its hf:// URL."""
        local = ROOT / rel
        return local.as_posix() if local.exists() else f"{HF_DATASET}/{rel}"


    TRAIN_PATH = src("processed/model_table/train.parquet")
    VAL_PATH = src("processed/model_table/validation.parquet")
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    DATA_PATHS = [TRAIN_PATH, VAL_PATH]   # hashed into each run's config when local
    print(f"train {len(train):,} rows ({train['target'].mean():.1%} positive) | validation {len(val):,} rows ({val['target'].mean():.1%} positive)")

    # Source: 05_baseline_models.ipynb, code cell 4
    # ---- 1. Naive rules (the bar, no ML) -------------------------------------------
    # Rates come from training rows only. The family rule uses the highest training rate among the
    # storm families present in the episode; the reference rule = family rate x county history / training rate.
    rules = naive_rules(train, val)
    REF = rules["family_x_history"]

    rows = []
    for name, p in rules.items():
        m, _ = evaluate(val, p, train)
        rows.append({"rule": name, **{k: m[k] for k in ["pr_auc", "roc_auc", "brier", "brier_skill", "recall_top10", "precision_top10"]}})
    naive = pd.DataFrame(rows)
    naive.to_csv(TABLES / "naive_rules_validation.csv", index=False)

    # First ad-hoc PR-AUC values (before the rules were code); only the reference rule differs, see success_criteria.md
    original = {"always_train_rate": 0.165, "family_rate": 0.312, "county_history": 0.368, "family_x_history": 0.472}
    naive["original_pr_auc"] = naive["rule"].map(original)
    print(naive.round(4))

    # Source: 05_baseline_models.ipynb, code cell 5
    # ---- 2. Log the naive rules to W&B ----------------------------------------------
    RULE_NOTES = {
        "always_train_rate": "constant = training positive rate",
        "family_rate": "max training outage rate among storm families present in the episode",
        "county_history": "hist_outage_rate (county past outage rate, shrunk to national; leave-one-year-out in train)",
        "family_x_history": "family_rate * county_history / training rate, clipped to [0, 1] (reference rule)",
    }
    if LOG_TO_WANDB:
        for name, p in rules.items():
            config = {"model": name, "model_type": "naive_rule", "rule": RULE_NOTES[name],
                      "train_rows": len(train), "train_pos_rate": train["target"].mean()}
            _, _, url = log_run(f"naive_{name}", "baseline", val, p, train, config,
                                p_ref=None if name == "family_x_history" else REF, data_paths=DATA_PATHS)
            print(name, "->", url)

    # Source: 05_baseline_models.ipynb, code cell 6
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    features = ["month", "local_hour", "pre_peak_24h", "pct_out_at_start"]

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(train[features], train["target"])

    print("Logistic regression trained.")

    # Source: 05_baseline_models.ipynb, code cell 7
    predictions = model.predict_proba(val[features])[:, 1]

    metrics, _ = evaluate(val, predictions, train)

    comparison = {
        "rule": "logistic_regression",
        **{
            key: metrics[key]
            for key in [
                "pr_auc", "roc_auc", "brier",
                "brier_skill", "recall_top10", "precision_top10"
            ]
        }
    }

    print(pd.concat(
        [naive.drop(columns="original_pr_auc"), pd.DataFrame([comparison])],
        ignore_index=True
    ).round(4))

    # Source: 05_baseline_models.ipynb, code cell 8
    results = pd.DataFrame([comparison])
    results["features"] = ", ".join(features)
    results["train_years"] = "2017–2023"
    results["validation_year"] = 2024

    results.to_csv(
        TABLES / "logistic_regression_initial_validation.csv",
        index=False
    )

    print("First model results saved.")

    # Source: 05_baseline_models.ipynb, code cell 9
    from sklearn.metrics import confusion_matrix

    # Initial diagnostic threshold, not our final alert threshold
    predicted_labels = predictions >= 0.5

    print(pd.DataFrame(
        confusion_matrix(val["target"], predicted_labels, labels=[0, 1]),
        index=["Actual: no major outage", "Actual: major outage"],
        columns=["Predicted: no major outage", "Predicted: major outage"]
    ))

    # Source: 05_baseline_models.ipynb, code cell 10
    from sklearn.metrics import confusion_matrix

    rows = []

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        alerts = predictions >= threshold
        tn, fp, fn, tp = confusion_matrix(
            val["target"], alerts, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "recall": tp / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "missed_outages": fn,
            "false_alarms": fp,
            "false_alarms_per_1000": 1000 * fp / len(val)
        })

    print(pd.DataFrame(rows).round(3))

    # Source: 05_baseline_models.ipynb, code cell 12
    pd.DataFrame(rows).to_csv(
        TABLES / "logistic_regression_threshold_comparison.csv",
        index=False
    )

    # Source: 05_baseline_models.ipynb, code cell 13
    # ---- 3. Log the logistic regression to W&B ----------------------------------------
    # Same checks as the naive rules; the bootstrap CI compares against the reference rule (family_x_history).
    if LOG_TO_WANDB:
        lr = model.named_steps["logisticregression"]
        config = {"model": "logistic_regression", "model_type": "logistic_regression",
                  "features": features, "n_features": len(features),
                  "preprocessing": "median impute + missing indicators, standard scaling (fit on train)",
                  "C": lr.C, "penalty": lr.penalty, "solver": lr.solver, "max_iter": lr.max_iter,
                  "class_weight": str(lr.class_weight),
                  "train_rows": len(train), "train_pos_rate": train["target"].mean()}
        _, _, url = log_run("logistic_regression_initial", "baseline", val, predictions, train, config,
                            p_ref=REF, data_paths=DATA_PATHS,
                            extra_tables={"threshold_comparison": pd.DataFrame(rows)})
        print("logistic_regression ->", url)


if __name__ == "__main__":
    main()
