# Shared evaluation + Weights & Biases logging for every model (step 05 onward).
# The checks follow reports/success_criteria.md, so every model is judged the same way.
# Nothing here uploads data: W&B only receives metrics, small summary tables and charts.
import hashlib
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
WANDB_ENTITY = "leslie-ortega-us-georgia-state-university"
WANDB_PROJECT = "CAPSTONE MODEL"
HF_DATASET = "DSCI-Capstone/county-outage-data"
HF_REVISION = "5c01325258aa922bf395da1e39acda7c234fefcf"  # dataset commit the model table was downloaded from

# Storm families as in the modeling table (n_<family> = report count in the episode)
FAMILIES = ["tstm_wind", "hail", "tornado", "high_wind", "flood", "winter", "ice_storm",
            "tropical", "cold", "lightning", "wildfire", "other"]
CHECK_FAMILIES = ["tstm_wind", "winter", "high_wind", "hail", "flood", "tornado"]  # subgroup diagnostics

# Reference-relative criteria revised 2026-10-27; see reports/success_criteria.md.
TOP_SHARE = 0.10
N_BOOT = 1000      # episode-level bootstrap resamples (provisional)
SEED = 42

BLUE, GRAY, SURFACE = "#2a78d6", "#a3a29c", "#fcfcfb"


# ---- Naive rules (the bar) ---------------------------------------------------
def family_rates(train):
    """Training outage rate for each storm family (rows where the family is present)."""
    return {f: train.loc[train[f"n_{f}"] > 0, "target"].mean() for f in FAMILIES}


def family_rule(train, frame):
    """Storm family rate: the highest training rate among the families present in the episode."""
    rates = family_rates(train)
    present = np.column_stack([np.where(frame[f"n_{f}"] > 0, rates[f], np.nan) for f in FAMILIES])
    return np.nanmax(present, axis=1)


def naive_rules(train, frame):
    """The four no-ML rules from success_criteria.md, as predicted probabilities for `frame`."""
    base = train["target"].mean()
    fam = family_rule(train, frame)
    hist = frame["hist_outage_rate"].to_numpy()
    return {
        "always_train_rate": np.full(len(frame), base),
        "family_rate": fam,
        "county_history": hist,
        "family_x_history": np.clip(fam * hist / base, 0, 1),
    }


# ---- Metrics -----------------------------------------------------------------
def recall_at_top(y, p, share=TOP_SHARE):
    """Share of all positives caught when flagging the top `share` of rows by predicted risk."""
    k = int(np.ceil(share * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return y[top].sum() / y.sum()


def precision_at_top(y, p, share=TOP_SHARE):
    k = int(np.ceil(share * len(y)))
    return y[np.argsort(-p, kind="stable")[:k]].mean()


def bootstrap_gain(y, p, p_ref, episodes, n_boot=N_BOOT, seed=SEED):
    """95% CI of PR-AUC(model) - PR-AUC(reference), resampling whole storm episodes."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(episodes)
    rows_by_ep = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate(rows_by_ep[rng.integers(0, len(uniq), len(uniq))])
        if y[idx].sum() == 0:
            continue
        gains.append(average_precision_score(y[idx], p[idx]) - average_precision_score(y[idx], p_ref[idx]))
    return np.percentile(gains, [2.5, 97.5])


def calibration_deciles(y, p):
    """Mean prediction vs observed rate in each predicted-probability decile (in points)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False) + 1
    t = d.groupby("decile").agg(rows=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    t["gap_pts"] = 100 * (t["mean_pred"] - t["observed"])
    return t


def group_pr_auc(y, p, p_ref, groups):
    """PR-AUC of the model and of the reference rule within each group (dict name -> boolean mask)."""
    out = []
    for name, mask in groups.items():
        if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            continue  # PR-AUC undefined without both classes
        m, f = average_precision_score(y[mask], p[mask]), average_precision_score(y[mask], p_ref[mask])
        out.append({"group": name, "rows": int(mask.sum()), "pos_rate": y[mask].mean(),
                    "pr_auc": m, "reference_pr_auc": f, "beats_reference": m > f})
    return pd.DataFrame(out)


def evaluate(frame, p, train, p_ref=None):
    """All success-criteria checks for predictions `p` on `frame` (validation).
    p_ref = reference-rule predictions; None computes the rule and skips the optional bootstrap."""
    y = frame["target"].to_numpy()
    p = np.asarray(p, dtype=float)
    run_bootstrap = p_ref is not None
    p_ref = np.asarray(naive_rules(train, frame)["family_x_history"] if p_ref is None else p_ref, dtype=float)

    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, np.full(len(y), y.mean()))   # always predicting the observed rate
    deciles = calibration_deciles(y, p)

    fam_groups = {f: (frame[f"n_{f}"] > 0).to_numpy() for f in CHECK_FAMILIES}
    region_groups = {f"FEMA {r}": (frame["fema_region"] == r).to_numpy() for r in sorted(frame["fema_region"].unique())}
    families = group_pr_auc(y, p, p_ref, fam_groups)
    regions = group_pr_auc(y, p, p_ref, region_groups)
    by_year = pd.DataFrame([{"year": yr, "rows": int(m.sum()), "pos_rate": y[m].mean(), "mean_pred": p[m].mean(),
                             "pr_auc": average_precision_score(y[m], p[m])}
                            for yr in sorted(frame["year"].unique()) for m in [(frame["year"] == yr).to_numpy()]])

    m = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier,
        "brier_skill": 1 - brier / brier_base,
        "cal_overall_pts": 100 * (p.mean() - y.mean()),
        "cal_worst_decile_pts": deciles["gap_pts"].abs().max(),
        "recall_top10": recall_at_top(y, p),
        "precision_top10": precision_at_top(y, p),
        "max_recall_top10": min(1.0, np.ceil(TOP_SHARE * len(y)) / y.sum()),
        "pos_rate": y.mean(),
        "rows": len(y),
    }
    if run_bootstrap:
        lo, hi = bootstrap_gain(y, p, np.asarray(p_ref, dtype=float), frame["episode_id"].to_numpy())
        m["pr_auc_gain_ci_low"], m["pr_auc_gain_ci_high"] = lo, hi

    # Compare full-precision predictions on the same rows, never rounded report values.
    m["criteria_version"] = "2026-10-27-reference"
    m["reference_pr_auc"] = average_precision_score(y, p_ref)
    m["reference_brier"] = brier_score_loss(y, p_ref)
    m["reference_brier_skill"] = 1 - m["reference_brier"] / brier_base
    m["reference_recall_top10"] = recall_at_top(y, p_ref)
    m["pass_1_ranking"] = bool(m["pr_auc"] > m["reference_pr_auc"])
    m["pass_2_brier_skill"] = bool(m["brier"] <= m["reference_brier"])
    m["pass_5_recall_top10"] = bool(m["recall_top10"] >= m["reference_recall_top10"])
    m["passes_all_validation"] = all(m[k] for k in m if k.startswith("pass_"))
    m = {k: float(v) if isinstance(v, np.floating) else v for k, v in m.items()}

    tables = {"calibration_deciles": deciles, "by_family": families, "by_fema_region": regions, "by_year": by_year}
    return m, tables


# ---- Charts --------------------------------------------------------------------
def calibration_plot(deciles, title):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = max(deciles["mean_pred"].max(), deciles["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GRAY, lw=1, label="perfect calibration")
    ax.plot(deciles["mean_pred"], deciles["observed"], "o-", color=BLUE, label="model (deciles)")
    ax.set(xlabel="mean predicted probability", ylabel="observed outage rate", title=title, xlim=(0, lim), ylim=(0, lim))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return fig


def pr_curve_plot(y, p, p_ref, title):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for pred, color, label in [(p_ref, GRAY, "reference rule"), (p, BLUE, "model")]:
        if pred is None:
            continue
        prec, rec, _ = precision_recall_curve(y, pred)
        ax.plot(rec, prec, color=color, label=f"{label} (PR-AUC {average_precision_score(y, pred):.3f})")
    ax.set(xlabel="recall", ylabel="precision", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return fig


# ---- W&B -----------------------------------------------------------------------
def data_version(paths):
    """What data a run used: the HF dataset commit plus a short hash of each local table file."""
    v = {"hf_dataset": HF_DATASET, "hf_revision": HF_REVISION}
    for p in paths:
        p = Path(p)
        if p.exists():
            v[f"md5_{p.stem}"] = hashlib.md5(p.read_bytes()).hexdigest()[:12]
    return v


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_run(name, job_type, frame, p, train, config, p_ref=None, data_paths=(), extra_tables=None):
    """Evaluate predictions, then log metrics, tables and charts to W&B as one run. Returns (metrics, tables, url).
    extra_tables = optional {name: DataFrame} of model-specific tables (e.g. threshold comparison) to log alongside."""
    import wandb

    metrics, tables = evaluate(frame, p, train, p_ref=p_ref)
    cfg = {**config, "git_commit": git_commit(), **data_version(data_paths), "eval_split": "validation_2024"}
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, job_type=job_type, config=cfg, reinit="finish_previous")
    run.summary.update(metrics)
    for tname, t in {**tables, **(extra_tables or {})}.items():
        run.log({tname: wandb.Table(dataframe=t)})
    y = frame["target"].to_numpy()
    for key, fig in [("calibration_plot", calibration_plot(tables["calibration_deciles"], f"Calibration: {name}")),
                     ("pr_curve", pr_curve_plot(y, np.asarray(p, float), None if p_ref is None else np.asarray(p_ref, float), f"PR curve: {name}"))]:
        run.log({key: wandb.Image(fig)})
        plt.close(fig)
    url = run.url
    run.finish()
    return metrics, tables, url
