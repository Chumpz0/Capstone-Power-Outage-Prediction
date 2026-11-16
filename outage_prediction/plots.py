"""Figure builders return figures; notebooks and export callers own their lifecycle."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from .config import WINDOW_HOURS

SIG_PCT = 0.01  # Event-level description; separate from the 5% modeling label.

BLUE, ORANGE, GRAY, SURFACE = "#2a78d6", "#eb6834", "#a3a29c", "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"


def set_style():
    """Apply the shared notebook/report style explicitly, never on import."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#d6d5d0",
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ebeae6",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def calibration_plot(deciles, title):
    """Plot mean_pred versus observed fractions from a calibration-decile table."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = max(deciles["mean_pred"].max(), deciles["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GRAY, lw=1, label="perfect calibration")
    ax.plot(
        deciles["mean_pred"],
        deciles["observed"],
        "o-",
        color=BLUE,
        label="model (deciles)",
    )
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed outage rate",
        title=title,
        xlim=(0, lim),
        ylim=(0, lim),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return fig


def pr_curve_plot(y, p, p_ref, title):
    """Plot precision/recall and average precision for model and optional reference scores."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for pred, color, label in [(p_ref, GRAY, "reference rule"), (p, BLUE, "model")]:
        if pred is None:
            continue
        prec, rec, _ = precision_recall_curve(y, pred)
        ax.plot(
            rec,
            prec,
            color=color,
            label=f"{label} (PR-AUC {average_precision_score(y, pred):.3f})",
        )
    ax.set(xlabel="recall", ylabel="precision", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return fig


import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def outage_state_counts(state_days):
    """Build the corresponding training-only event-level EDA figure."""
    top = state_days.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["state"], top["sig_outage_days"] / 1000, color=BLUE, height=0.7)
    ax.set_xlabel("County-days with >= 1% of customers out (thousands)")
    ax.set_title("Outage count by state, top 20 (2017-2023)")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(top["sig_outage_days"] / 1000):
        ax.text(v, y, f" {v:.1f}k", va="center", color=INK_2, fontsize=8)
    return fig


def outage_state_rates(state_days):
    """Build the corresponding training-only event-level EDA figure."""
    # Counts favour states with many counties; this rate removes that effect.
    # Territories (PR, VI, GU, AS) have chronic grid outages that dwarf every state,
    # so they are listed in a note instead of plotted.
    TERRITORIES = ["PR", "VI", "GU", "AS", "MP"]
    terr = state_days[state_days["state"].isin(TERRITORIES)]
    terr_note = ", ".join(
        f"{s} {v:.0f}" for s, v in zip(terr["state"], terr["sig_days_per_county_year"])
    )
    top = (
        state_days[~state_days["state"].isin(TERRITORIES)]
        .sort_values("sig_days_per_county_year", ascending=False)
        .head(20)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["state"], top["sig_days_per_county_year"], color=BLUE, height=0.7)
    ax.set_xlabel("Days per county per year with >= 1% of customers out")
    ax.set_title("Outage rate by state, top 20 (2017-2023)")
    ax.text(
        0.98,
        0.02,
        f"Not shown (territories): {terr_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=INK_2,
        fontsize=8,
    )
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(top["sig_days_per_county_year"]):
        ax.text(v, y, f" {v:.1f}", va="center", color=INK_2, fontsize=8)
    return fig


def storm_type_counts(by_type):
    """Build the corresponding training-only event-level EDA figure."""
    top = by_type.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top["outages"] / 1000, color=BLUE, height=0.7)
    ax.set_xlabel("Storm-county rows followed by >= 1% out within 24 h (thousands)")
    ax.set_title("Storm outage count by storm type (2017-2023)")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(top["outages"] / 1000):
        ax.text(v, y, f" {v:.1f}k", va="center", color=INK_2, fontsize=8)
    return fig


def storm_type_rates(by_type):
    """Build the corresponding training-only event-level EDA figure."""
    # Only storm types with at least 500 storm-county rows, so rates are stable.
    rate = by_type[by_type["rows"] >= 500].sort_values("outage_rate").tail(15)
    y = np.arange(len(rate))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(
        y + 0.2,
        rate["outage_rate"] * 100,
        height=0.38,
        color=BLUE,
        label="24 h after storm start",
    )
    ax.barh(
        y - 0.2,
        rate["baseline_rate"] * 100,
        height=0.38,
        color=GRAY,
        label="Same hours, one week earlier",
    )
    ax.set_yticks(y, rate.index)
    ax.set_xlabel("% of storm-county rows with >= 1% of customers out")
    ax.set_title("Chance of an outage by storm type (2017-2023)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    for yy, v in zip(y, rate["outage_rate"] * 100):
        ax.text(v, yy + 0.2, f" {v:.0f}%", va="center", color=INK_2, fontsize=8)
    return fig


def wind_probability(wind_rate):
    """Build the corresponding training-only event-level EDA figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        wind_rate.index.astype(str),
        wind_rate["outage_rate"] * 100,
        color=BLUE,
        width=0.7,
    )
    ax.set_xlabel("Reported wind speed (knots)")
    ax.set_ylabel("% with >= 1% customers out")
    ax.set_title("Outage chance by reported wind speed (thunderstorm + high wind)")
    ax.grid(axis="x", visible=False)
    for x, (v, n) in enumerate(zip(wind_rate["outage_rate"] * 100, wind_rate["rows"])):
        ax.text(
            x,
            v,
            f"{v:.0f}%\nn={n:,}",
            ha="center",
            va="bottom",
            color=INK_2,
            fontsize=8,
        )
    ax.set_ylim(0, 100)
    return fig


def severity_exceedance(peaks):
    """Build the corresponding training-only event-level EDA figure."""
    # For each threshold X: share of storm-county rows whose 24 h peak reached X% out.
    # This shows how the target threshold choice changes the positive rate.
    thresholds = np.logspace(-3, np.log10(0.5), 60)
    storm_curve = [(peaks["peak_24h"] >= t).mean() * 100 for t in thresholds]
    base_curve = [(peaks["baseline_24h"] >= t).mean() * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        thresholds * 100,
        storm_curve,
        color=BLUE,
        linewidth=2,
        label="24 h after storm start",
    )
    ax.plot(
        thresholds * 100,
        base_curve,
        color=GRAY,
        linewidth=2,
        label="Same hours, one week earlier",
    )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 5, 10, 50], ["0.1%", "0.5%", "1%", "5%", "10%", "50%"])
    ax.set_xlabel("Threshold: % of county customers out (log scale)")
    ax.set_ylabel("% of storm-county rows at or above it")
    ax.set_title("How often storms reach each outage level (2017-2023)")
    for t in [0.01, 0.05, 0.10]:
        v = (peaks["peak_24h"] >= t).mean() * 100
        ax.plot(
            t * 100,
            v,
            "o",
            color=BLUE,
            markersize=7,
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
        )
        ax.annotate(
            f"{t:.0%}: {v:.0f}%",
            (t * 100, v),
            textcoords="offset points",
            xytext=(8, 6),
            color=INK_2,
            fontsize=8,
        )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    return fig


def peak_timing(peaks):
    """Build the corresponding training-only event-level EDA figure."""
    # Rows that reached >= 1% within 48 h; the hour (0-47) of their worst reading.
    hit = peaks[peaks["peak_48h"] >= SIG_PCT]
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = hit["hour_of_peak"].value_counts().sort_index()
    ax.bar(counts.index, counts.values / 1000, color=BLUE, width=0.8)
    ax.axvline(WINDOW_HOURS - 0.5, color=INK_2, linestyle="--", linewidth=1)
    share_24 = (hit["hour_of_peak"] < WINDOW_HOURS).mean()
    ax.text(
        WINDOW_HOURS,
        ax.get_ylim()[1] * 0.9,
        f"  {share_24:.0%} peak within 24 h",
        color=INK_2,
        fontsize=9,
    )
    ax.set_xlabel("Hours after storm start")
    ax.set_ylabel("Storm-county rows (thousands)")
    ax.set_title("Hour of peak outage after a storm starts (2017-2023)")
    ax.grid(axis="x", visible=False)
    return fig


def outage_seasonality(monthly):
    """Build the corresponding training-only event-level EDA figure."""
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(month_names, monthly["sig_days_per_year"] / 1000, color=BLUE, width=0.7)
    ax.set_ylabel("County-days with >= 1% out\n(thousands, average per year)")
    ax.set_title("Outages by month (2017-2023)")
    ax.grid(axis="x", visible=False)
    return fig


def event_figures(tables):
    """Return event-level figures without writing files or closing caller-owned figures."""
    return {
        "01_outage_days_by_state": outage_state_counts(tables["state_days"]),
        "02_outage_rate_by_state": outage_state_rates(tables["state_days"]),
        "03_storm_outage_count_by_type": storm_type_counts(tables["by_type"]),
        "04_outage_probability_by_type": storm_type_rates(tables["by_type"]),
        "05_wind_speed_vs_outage": wind_probability(tables["wind_rate"]),
        "06_outage_severity_exceedance": severity_exceedance(tables["peaks"]),
        "07_hours_to_peak": peak_timing(tables["peaks"]),
        "08_outages_by_month": outage_seasonality(tables["monthly"]),
    }


def target_profiles(train, by_year):
    """Plot split-year prevalence and training-only monthly/regional target profiles."""
    BASE = train.target.mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    yr = by_year.dropna(subset=["positive_rate"])
    ax.bar(
        yr["year"].astype(str),
        100 * yr["positive_rate"],
        color=[BLUE] * (len(yr) - 1) + [GRAY],
    )
    ax.axhline(100 * BASE, color=INK_2, lw=1, ls="--")
    ax.set(title="Positive rate by year", ylabel="% of county-episodes >= 5% out")
    ax.text(
        len(yr) - 1,
        100 * yr["positive_rate"].iloc[-1] + 0.4,
        "validation",
        ha="center",
        fontsize=8,
        color=INK_2,
    )

    ax = axes[1]
    m = train.groupby("month")["target"].mean()
    ax.bar(m.index, 100 * m.values, color=BLUE)
    ax.axhline(100 * BASE, color=INK_2, lw=1, ls="--")
    ax.set(title="By month (train)", xticks=range(1, 13), ylabel="% positive")

    ax = axes[2]
    r = train.groupby("fema_region")["target"].agg(["mean", "size"])
    ax.bar(r.index.astype(str), 100 * r["mean"], color=BLUE)
    ax.axhline(100 * BASE, color=INK_2, lw=1, ls="--")
    ax.set(title="By FEMA region (train)", xlabel="FEMA region", ylabel="% positive")
    fig.tight_layout()
    return fig


def peak_distribution(train):
    """Plot the training outcome distribution with the fixed 5% label threshold."""
    peak = train["retro_peak_24h"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(peak[peak > 0]), bins=60, color=BLUE)
    ax.axvline(np.log10(0.05), color=ORANGE, lw=1.5)
    ax.text(
        np.log10(0.05) + 0.05,
        ax.get_ylim()[1] * 0.9,
        "target threshold 5%",
        color=ORANGE,
        fontsize=9,
    )
    ax.set(
        title="Peak share of customers out within 24 h (train, log scale)",
        xlabel="log10(peak share out)",
        ylabel="county-episodes",
    )
    return fig


def univariate_signal(signal):
    """Plot descriptive one-feature training associations."""
    fig, ax = plt.subplots(figsize=(8, 9))
    s = signal.iloc[::-1]
    ax.barh(
        s["feature"],
        s["strength"] - 0.5,
        left=0.5,
        color=[
            BLUE if d.startswith("higher -> more") else ORANGE for d in s["direction"]
        ],
    )
    ax.axvline(0.5, color=GRAY, lw=1)
    ax.set(
        title="Signal of each feature alone (train ROC-AUC)",
        xlabel="ROC-AUC as a single score (strength)",
        xlim=(0.5, None),
    )
    ax.text(
        0.99,
        0.02,
        "blue: higher value -> more outages\norange: higher value -> fewer outages",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
    )
    return fig


def storm_profiles(fam, loc, pre_t, base):
    """Plot training family, mapping and pre-storm condition profiles."""
    BASE = base
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1.4, 1, 1]}
    )
    ax = axes[0]
    ax.barh(fam["family"], 100 * fam["rate"], color=BLUE)
    for i, (rt, n) in enumerate(zip(fam["rate"], fam["rows"])):
        ax.text(100 * rt + 0.8, i, f"{n:,}", va="center", fontsize=8, color=INK_2)
    ax.axvline(100 * BASE, color=GRAY, lw=1, ls="--")
    ax.set(
        title="Outage rate when the family is present",
        xlabel="% positive (label = rows)",
    )
    ax = axes[1]
    ax.bar(
        range(len(loc)), 100 * loc["rate"], color=[BLUE, BLUE, BLUE, GRAY][: len(loc)]
    )
    ax.set_xticks(
        range(len(loc)),
        [t.replace(", ", ",\n").replace(" only", "\nonly") for t in loc.index],
        fontsize=8,
    )
    ax.axhline(100 * BASE, color=GRAY, lw=1, ls="--")
    ax.set(title="How NOAA located the storm", ylabel="% positive")
    ax = axes[2]
    ax.bar(
        range(len(pre_t)), 100 * pre_t["rate"], color=[BLUE] * (len(pre_t) - 1) + [GRAY]
    )
    ax.set_xticks(
        range(len(pre_t)),
        [t.replace(" (day before unobserved)", "") for t in pre_t.index],
        fontsize=8,
    )
    ax.axhline(100 * BASE, color=GRAY, lw=1, ls="--")
    ax.set(title="Share out in the hour before the start", ylabel="% positive")
    fig.tight_layout()
    return fig


def feature_correlations(corr):
    """Plot the prepared training Spearman correlation matrix."""
    FEATURES = list(corr.columns)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#fcfcfb", BLUE])
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Feature-feature correlation (train)")
    return fig


def drift_comparison(drift):
    """Plot observed feature-distribution drift; unavailable comparisons stay missing."""
    top = drift[~drift["feature"].eq("max_hurricane_cat")].head(12).iloc[::-1]
    XMAX = 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(top))
    ax.barh(
        yy + 0.2,
        top["psi_validation"].clip(upper=XMAX),
        height=0.4,
        color=GRAY,
        label="validation 2024",
    )
    ax.barh(
        yy - 0.2,
        top["psi_test"].clip(upper=XMAX),
        height=0.4,
        color=BLUE,
        label="test 2025 (features only)",
    )
    for i, (pv, pt) in enumerate(zip(top["psi_validation"], top["psi_test"])):
        if max(pv, pt) > XMAX:
            ax.text(
                XMAX * 0.98,
                i,
                f"PSI {pv:.1f} / {pt:.1f}  ",
                ha="right",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax.set_yticks(yy, top["feature"])
    ax.axvline(0.1, color=INK_2, lw=0.8, ls=":")
    ax.axvline(0.25, color=ORANGE, lw=0.8, ls=":")
    ax.set(
        title="Feature drift vs training years (PSI, top 12)",
        xlabel="PSI (axis capped at 0.5; dotted: 0.1 moderate, 0.25 large)",
        xlim=(0, XMAX),
    )
    ax.legend(loc="center right")
    return fig


def split_distributions(train, val, test):
    """Plot selected split-feature distributions; no test outcomes are required."""
    from .analysis import as_num

    DIST = [
        "n_events",
        "county_episode_hours",
        "max_wind_kts",
        "max_hail_in",
        "episode_counties",
        "hist_outage_rate",
        "log_customers",
        "pct_customers_tracked",
        "local_hour",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, col in zip(axes.flat, DIST):
        data = [as_num(d[col]).dropna() for d in (train, val, test)]
        lo, hi = np.nanquantile(pd.concat(data), [0.0, 0.99])
        edges = np.linspace(lo, hi, 40)
        for d, color, lab in zip(
            data, [BLUE, GRAY, ORANGE], ["train", "validation", "test"]
        ):
            ax.hist(
                d.clip(upper=hi),
                bins=edges,
                density=True,
                histtype="step",
                lw=1.5,
                color=color,
                label=lab,
            )
        ax.set_title(col, fontsize=10)
    axes.flat[0].legend()
    fig.text(
        0.01,
        0.005,
        "values above the 99th percentile are shown in the last bin",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def feature_bin_profiles(train, curves):
    """Plot prepared training outage rates with their bin sizes and missing-value reference."""
    from .analysis import CURVES

    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, col in zip(axes.flat, CURVES):
        rows = curves.loc[curves.feature.eq(col)]
        ax.plot(range(len(rows)), 100 * rows.rate, "o-", color=BLUE, ms=4)
        ax.set_xticks(
            range(len(rows)), [f"{v:.2g}" for v in rows.x], rotation=45, fontsize=8
        )
        ax.axhline(100 * train.target.mean(), color=GRAY, ls="--", lw=1)
        if train[col].isna().mean() > 0.01:
            ax.axhline(
                100 * train.loc[train[col].isna(), "target"].mean(),
                color=ORANGE,
                ls=":",
                lw=1,
            )
        ax.set(title=col, ylabel="% positive", xlabel="bin median")
    fig.tight_layout()
    return fig


def model_figures(train, validation, test, tables):
    """Return the eight ML-dataset EDA figures; caller chooses inline display/export."""
    return {
        "01_target_by_year_month_region": target_profiles(
            train, tables["rows_by_year"]
        ),
        "02_peak_share_distribution": peak_distribution(train),
        "03_univariate_signal": univariate_signal(tables["univariate_signal"]),
        "04_rate_by_feature_bins": feature_bin_profiles(
            train, tables["rate_by_feature_bins"]
        ),
        "05_family_coding_prestorm": storm_profiles(
            tables["rate_by_storm_family"],
            tables["rate_by_storm_coding"],
            tables["rate_by_pre_storm_outage"],
            train.target.mean(),
        ),
        "06_feature_correlation": feature_correlations(tables["feature_correlation"]),
        "07_feature_drift_psi": drift_comparison(tables["feature_drift_psi"]),
        "08_feature_distributions_by_split": split_distributions(
            train, validation, test
        ),
    }
