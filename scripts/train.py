"""Train one baseline model on 2017-2023, score 2024 validation, and log the run to W&B.

Examples (run from the repo root):
    python scripts/train.py --model logreg --C 0.1
    python scripts/train.py --model hgb --learning-rate 0.05 --max-leaf-nodes 63
    python scripts/train.py --model hgb --no-wandb          # evaluate locally, send nothing

Every run is judged with the same success-criteria checks (src/evaluation.py) as notebook 05.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_model_table  # noqa: E402
from src.models import MODELS, run_experiment  # noqa: E402

SHOW = ["pr_auc", "pr_auc_gain_ci_low", "pr_auc_gain_ci_high", "roc_auc", "log_loss", "brier", "brier_skill",
        "cal_overall_pts", "cal_worst_decile_pts", "recall_top10", "precision_top10", "fit_seconds", "predict_seconds"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--name", help="W&B run name (default: built from the model and its settings)")
    ap.add_argument("--notes", default="", help="free-text note stored with the run")
    ap.add_argument("--no-wandb", action="store_true", help="evaluate locally only; nothing is sent to W&B")
    ap.add_argument("--class-weight", choices=["balanced"], default=None)
    lr = ap.add_argument_group("logistic regression")
    lr.add_argument("--C", type=float, default=1.0, help="inverse regularisation strength")
    hgb = ap.add_argument_group("gradient boosting")
    hgb.add_argument("--learning-rate", type=float, default=0.1)
    hgb.add_argument("--max-iter", type=int, default=500)
    hgb.add_argument("--max-leaf-nodes", type=int, default=31)
    hgb.add_argument("--min-samples-leaf", type=int, default=20)
    hgb.add_argument("--l2", type=float, default=0.0, help="L2 regularisation")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.model == "logreg":
        params = {"C": args.C}
    else:
        params = {"learning_rate": args.learning_rate, "max_iter": args.max_iter, "max_leaf_nodes": args.max_leaf_nodes,
                  "min_samples_leaf": args.min_samples_leaf, "l2_regularization": args.l2}
    if args.class_weight:
        params["class_weight"] = args.class_weight

    train, val = load_model_table()
    result = run_experiment(args.model, params, train, val, name=args.name, log=not args.no_wandb, notes=args.notes)

    m = result["metrics"]
    print(f"\n{result['name']}")
    for k in SHOW:
        if k in m:
            print(f"  {k:22s} {m[k]:.4f}")
    passed = [k for k in m if k.startswith("pass_")]
    print("  criteria passed:      ", ", ".join(k for k in passed if m[k]) or "none")
    print("  criteria failed:      ", ", ".join(k for k in passed if not m[k]) or "none")
    if result["url"]:
        print(f"  W&B run: {result['url']}")


if __name__ == "__main__":
    main()
