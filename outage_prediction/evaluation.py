"""Training-derived reference rules and explicit split scoring; no file or network I/O."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from .config import (
    FAMILIES,
    CHECK_FAMILIES,
    STORM_FEATURES,
    FEATURES,
    TOP_SHARE,
    N_BOOT,
    SEED,
    TRAIN_FIRST,
    TRAIN_LAST,
    VAL_YEAR,
    TEST_YEAR,
)


def validate_frame(frame, name, features=None):
    """Fail closed on wrong years, malformed labels, or unusable features."""
    features = FEATURES if features is None else features
    required = features + ["year", "target", "fips", "episode_id"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if name not in {"train", "validation", "test"}:
        raise ValueError("Expected train, validation or test split")
    allowed = range(TRAIN_FIRST, TRAIN_LAST + 1) if name == "train" else [
        VAL_YEAR if name == "validation" else TEST_YEAR
    ]
    if frame.empty or not frame["year"].isin(allowed).all():
        raise ValueError(
            f"{name}: expected only years {list(allowed)} and nonempty data"
        )
    if frame["target"].isna().any() or set(frame["target"].unique()) != {0, 1}:
        raise ValueError(
            f"{name}: target must contain both binary classes, without nulls"
        )
    if (
        frame[["fips", "episode_id"]].isna().any().any()
        or frame.duplicated(["fips", "episode_id"]).any()
    ):
        raise ValueError(f"{name}: county/episode keys must be present and unique")
    validate_features(frame, features, name)


def validate_features(frame, features, name="features"):
    """Validate numeric model inputs independently of labels, including unlabeled test rows."""
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    for feature in features:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"{name}: {feature} must be numeric")
        values = frame[feature].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or np.isnan(values).all():
            raise ValueError(f"{name}: {feature} is infinite or entirely missing")


def validate_probabilities(frame, predictions, name="predictions"):
    """Reject misaligned Series, wrong lengths, and nonfinite/out-of-range scores."""
    if isinstance(predictions, pd.Series) and not predictions.index.equals(frame.index):
        raise ValueError(f"{name}: prediction index does not match input rows")
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError(f"{name}: expected one finite probability per row")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"{name}: probabilities must be in [0, 1]")
    return values


def family_rates(train):
    """Training outage rate for each storm family (rows where the family is present)."""
    return {f: train.loc[train[f"n_{f}"] > 0, "target"].mean() for f in FAMILIES}


def family_rule(train, frame):
    """Storm family rate: the highest training rate among the families present in the episode."""
    rates = family_rates(train)
    present = np.column_stack(
        [np.where(frame[f"n_{f}"] > 0, rates[f], np.nan) for f in FAMILIES]
    )
    if np.isnan(present).all(axis=1).any():
        raise ValueError(
            "Reference rule: no supported training family for one or more rows"
        )
    return np.nanmax(present, axis=1)


def naive_rules(train, frame):
    """The four no-ML rules in docs/evaluation.md, as probabilities for `frame`."""
    base = train["target"].mean()
    if not 0 < base < 1:
        raise ValueError("Reference training labels must contain both classes")
    fam = family_rule(train, frame)
    hist = validate_probabilities(frame, frame["hist_outage_rate"], "county history")
    return {
        "always_train_rate": np.full(len(frame), base),
        "family_rate": fam,
        "county_history": hist,
        "family_x_history": np.clip(fam * hist / base, 0, 1),
    }


def recall_at_top(y, p, share=TOP_SHARE):
    """Share of all positives caught when flagging the top `share` of rows by predicted risk."""
    k = int(np.ceil(share * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return y[top].sum() / y.sum()


def precision_at_top(y, p, share=TOP_SHARE):
    """Positive fraction among the highest-risk rows, retaining input order for ties."""
    k = int(np.ceil(share * len(y)))
    return y[np.argsort(-p, kind="stable")[:k]].mean()


def bootstrap_gain(y, p, p_ref, episodes, n_boot=N_BOOT, seed=SEED):
    """95% CI of PR-AUC(model) - PR-AUC(reference), resampling whole storm episodes."""
    if n_boot < 1 or pd.isna(episodes).any():
        raise ValueError(
            "Bootstrap requires positive resample count and nonmissing episode IDs"
        )
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(episodes)
    rows_by_ep = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate(rows_by_ep[rng.integers(0, len(uniq), len(uniq))])
        if y[idx].sum() == 0:
            continue
        gains.append(
            average_precision_score(y[idx], p[idx])
            - average_precision_score(y[idx], p_ref[idx])
        )
    if not gains:
        raise ValueError("No bootstrap sample contained positive labels")
    return np.percentile(gains, [2.5, 97.5])


def calibration_deciles(y, p):
    """Mean prediction vs observed rate in each predicted-probability decile (in points)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False) + 1
    t = (
        d.groupby("decile")
        .agg(rows=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    t["gap_pts"] = 100 * (t["mean_pred"] - t["observed"])
    return t


def group_pr_auc(y, p, p_ref, groups):
    """PR-AUC of the model and of the reference rule within each group (dict name -> boolean mask)."""
    out = []
    for name, mask in groups.items():
        if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            continue  # PR-AUC undefined without both classes
        m, f = (
            average_precision_score(y[mask], p[mask]),
            average_precision_score(y[mask], p_ref[mask]),
        )
        out.append(
            {
                "group": name,
                "rows": int(mask.sum()),
                "pos_rate": y[mask].mean(),
                "pr_auc": m,
                "reference_pr_auc": f,
                "beats_reference": m > f,
            }
        )
    return pd.DataFrame(
        out,
        columns=[
            "group",
            "rows",
            "pos_rate",
            "pr_auc",
            "reference_pr_auc",
            "beats_reference",
        ],
    )


def evaluate(frame, p, train, p_ref=None, *, bootstrap=False, n_boot=N_BOOT,
             split="validation"):
    """Return (scalar metrics, diagnostic DataFrames) for labeled, keyed rows.

    Predictions are positional arrays or Series with exactly matching indexes.
    Both target classes, finite [0,1] probabilities and evaluation columns are
    required. The reference is training-derived; bootstrap is always explicit.
    Tied top-risk scores retain input row order. Brier skill is normalized by
    evaluation prevalence, as described in docs/evaluation.md.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    validate_frame(train, "train", [])
    validate_frame(frame, split, [])
    required = STORM_FEATURES + ["hist_outage_rate", "fema_region"]
    for name, data in [("train", train), (split, frame)]:
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"{name}: missing evaluation columns {sorted(missing)}")
    if set(train.episode_id) & set(frame.episode_id):
        raise ValueError(f"Training and {split} share storm episodes")
    y = frame["target"].to_numpy()
    p = validate_probabilities(frame, p)
    run_bootstrap = bootstrap
    p_ref = validate_probabilities(
        frame,
        naive_rules(train, frame)["family_x_history"] if p_ref is None else p_ref,
        "reference",
    )

    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(
        y, np.full(len(y), y.mean())
    )  # always predicting the observed rate
    deciles = calibration_deciles(y, p)

    fam_groups = {f: (frame[f"n_{f}"] > 0).to_numpy() for f in CHECK_FAMILIES}
    region_groups = {
        f"FEMA {r}": (frame["fema_region"] == r).to_numpy()
        for r in sorted(frame["fema_region"].unique())
    }
    families = group_pr_auc(y, p, p_ref, fam_groups)
    regions = group_pr_auc(y, p, p_ref, region_groups)
    by_year = pd.DataFrame(
        [
            {
                "year": yr,
                "rows": int(m.sum()),
                "pos_rate": y[m].mean(),
                "mean_pred": p[m].mean(),
                "pr_auc": average_precision_score(y[m], p[m]),
            }
            for yr in sorted(frame["year"].unique())
            for m in [(frame["year"] == yr).to_numpy()]
        ]
    )

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
        lo, hi = bootstrap_gain(
            y,
            p,
            np.asarray(p_ref, dtype=float),
            frame["episode_id"].to_numpy(),
            n_boot=n_boot,
        )
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
    m[f"passes_all_{split}"] = all(m[k] for k in m if k.startswith("pass_"))
    m = {k: float(v) if isinstance(v, np.floating) else v for k, v in m.items()}

    tables = {
        "calibration_deciles": deciles,
        "by_family": families,
        "by_fema_region": regions,
        "by_year": by_year,
    }
    return m, tables
