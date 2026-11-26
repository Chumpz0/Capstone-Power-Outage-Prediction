"""Project policy, compatible data resolution, and validated build publication.

Local paths are independent of the hosted dataset's historical directory names.
No import downloads data, creates directories, or configures credentials.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = "DSCI-Capstone/county-outage-data"
HF_REVISION = "5c01325258aa922bf395da1e39acda7c234fefcf"
TRAIN_FIRST, TRAIN_LAST, VAL_YEAR, TEST_YEAR = 2017, 2023, 2024, 2025
TARGET_PCT, WINDOW_HOURS, HISTORY_PRIOR = 0.05, 24, 20
MIN_DAILY_SNAPSHOT_PCT = 0.90
TOP_SHARE, N_BOOT, SEED = 0.10, 1000, 42
FAMILIES = [
    "tstm_wind",
    "hail",
    "tornado",
    "high_wind",
    "flood",
    "winter",
    "ice_storm",
    "tropical",
    "cold",
    "lightning",
    "wildfire",
    "other",
]
CHECK_FAMILIES = ["tstm_wind", "winter", "high_wind", "hail", "flood", "tornado"]
BASE_FEATURES = ["month", "local_hour", "pre_peak_24h", "pct_out_at_start"]
STORM_FEATURES = [f"n_{family}" for family in FAMILIES]
HISTORY_FEATURES = ["hist_episodes", "hist_outage_rate", "hist_mean_peak"]
FEATURES = BASE_FEATURES + STORM_FEATURES + HISTORY_FEATURES
FEATURE_SET = "base_storm_history"
WANDB_ENTITY = "leslie-ortega-us-georgia-state-university"
WANDB_PROJECT = "CAPSTONE MODEL"


def sha256(path):
    """Hash in bounded memory; only explicit final scoring may hash holdout labels."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    """Return the current commit without treating uncommitted changes as committed."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


@dataclass(frozen=True)
class DataSource:
    """A resolved reader path and provenance; local hashes do not verify a Hub revision."""

    path: str
    logical_path: str
    revision: str | None = None

    def metadata(self):
        """Describe the actual source, hashing local non-holdout files only."""
        result = {"path": self.path, "logical_path": self.logical_path}
        if self.revision:
            result.update(hf_dataset=HF_DATASET, hf_revision=self.revision)
        elif Path(self.path).is_file():
            if "test_labels" in self.logical_path or "holdout/" in self.logical_path:
                raise ValueError(
                    "Sealed labels must not be opened for provenance checks"
                )
            result.update(sha256=sha256(self.path), hf_revision_verified=False)
        return result


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve historical logical names against new local, old local, then pinned Hub paths."""

    root: Path = ROOT
    revision: str = HF_REVISION
    offline: bool = False

    @property
    def interim(self):
        return self.root / "data/interim"

    @property
    def processed(self):
        return self.root / "data/processed"

    @property
    def models(self):
        return self.root / "models"

    @property
    def cache(self):
        return self.root / ".cache"

    def candidates(self, logical):
        """Return new and legacy locations without touching file contents."""
        logical = str(logical).replace("\\", "/")
        rel = Path(logical)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Expected a project-relative logical data path")
        if logical.startswith("processed/model_table/"):
            new = "data/processed/" + logical.removeprefix("processed/model_table/")
        elif logical.startswith("processed/"):
            new = "data/interim/" + logical.removeprefix("processed/")
        elif logical.startswith("data/01_raw/"):
            new = "data/raw/" + logical.removeprefix("data/01_raw/")
        elif logical.startswith("data/02_combined/"):
            new = "data/raw/combined/" + logical.removeprefix("data/02_combined/")
        elif logical.startswith("data/reference/"):
            new = "data/external/" + logical.removeprefix("data/reference/")
        elif logical.startswith("holdout/"):
            new = "data/" + logical
        else:
            new = logical
        return self.root / new, self.root / logical

    def resolve(self, logical, *, allow_holdout=False):
        """Resolve without remote I/O. Reader errors never trigger an unpinned fallback."""
        logical = str(logical).replace("\\", "/")
        if (
            "test_labels" in logical or logical.startswith("holdout/")
        ) and not allow_holdout:
            raise ValueError("Test labels are sealed until final evaluation")
        for candidate in self.candidates(logical):
            if candidate.exists():
                return DataSource(candidate.as_posix(), logical)
        if self.offline:
            raise FileNotFoundError(f"No local copy of {logical}")
        if not self.revision or self.revision in {"main", "HEAD"}:
            raise ValueError("A fixed dataset revision is required for remote reads")
        return DataSource(
            f"hf://datasets/{HF_DATASET}@{self.revision}/{logical}",
            logical,
            self.revision,
        )


def data_version(paths):
    """Describe resolved sources without attributing a historical revision to local files."""
    sources = []
    for item in paths:
        source = (
            item if isinstance(item, DataSource) else DataSource(str(item), str(item))
        )
        sources.append(source.metadata())
    return {"sources": sources}


def require_checks(checks, *, name="dataset"):
    """Reject failed required invariants; diagnostic tables must be kept separate."""
    failed = [label for label, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(f"{name}: failed checks: {', '.join(failed)}")


@contextmanager
def staged_directory(destination):
    """Publish a validated directory with rollback, retaining the previous generation.

    Write and validate inside this context. A raised exception leaves the current
    destination intact. The fixed previous directory also supports crash recovery.
    No recursive operation is allowed outside the destination's checked parent.
    """
    destination = Path(destination).resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    previous = parent / (destination.name + ".previous")
    if previous.exists() and not destination.exists():
        rename_directory(previous, destination)
    stage = Path(
        tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=parent)
    ).resolve()
    if stage.parent != parent or previous.parent != parent:
        raise ValueError("Unsafe build paths")
    try:
        yield stage
        if previous.exists():
            shutil.rmtree(previous)
        had_previous = destination.exists()
        if had_previous:
            rename_directory(destination, previous)
        try:
            rename_directory(stage, destination)
        except OSError:
            if had_previous and not destination.exists():
                rename_directory(previous, destination)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def rename_directory(source, destination):
    """Retry transient Windows scanner locks, without hiding persistent publication failures."""
    for attempt in range(6):
        try:
            return source.rename(destination)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def export_results(tables, figures=(), directory=None):
    """Export analysis only when a caller supplies a destination; otherwise do nothing."""
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=True)
    for name, figure in dict(figures).items():
        figure.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")


# Geography and storm taxonomy are shared policy; local hours use majority standard time.
DROPPED_TYPES = ["Drought", "Heat", "Excessive Heat"]

STATE_UTC_OFFSET = {
    "CT": -5,
    "DE": -5,
    "DC": -5,
    "FL": -5,
    "GA": -5,
    "IN": -5,
    "KY": -5,
    "ME": -5,
    "MD": -5,
    "MA": -5,
    "MI": -5,
    "NH": -5,
    "NJ": -5,
    "NY": -5,
    "NC": -5,
    "OH": -5,
    "PA": -5,
    "RI": -5,
    "SC": -5,
    "VT": -5,
    "VA": -5,
    "WV": -5,
    "AL": -6,
    "AR": -6,
    "IL": -6,
    "IA": -6,
    "KS": -6,
    "LA": -6,
    "MN": -6,
    "MS": -6,
    "MO": -6,
    "NE": -6,
    "ND": -6,
    "OK": -6,
    "SD": -6,
    "TN": -6,
    "TX": -6,
    "WI": -6,
    "AZ": -7,
    "CO": -7,
    "ID": -7,
    "MT": -7,
    "NM": -7,
    "UT": -7,
    "WY": -7,
    "CA": -8,
    "NV": -8,
    "OR": -8,
    "WA": -8,
}

TYPE_GROUPS = {
    "tstm_wind": ["Thunderstorm Wind"],
    "hail": ["Hail"],
    "tornado": ["Tornado", "Funnel Cloud"],
    "high_wind": ["High Wind", "Strong Wind"],
    "flood": [
        "Flash Flood",
        "Flood",
        "Heavy Rain",
        "Coastal Flood",
        "Lakeshore Flood",
        "Storm Surge/Tide",
        "Debris Flow",
    ],
    "winter": [
        "Winter Weather",
        "Winter Storm",
        "Heavy Snow",
        "Blizzard",
        "Lake-Effect Snow",
        "Sleet",
    ],
    "ice_storm": ["Ice Storm"],
    "tropical": ["Tropical Storm", "Hurricane (Typhoon)", "Tropical Depression"],
    "cold": ["Extreme Cold/Wind Chill", "Cold/Wind Chill", "Frost/Freeze"],
    "lightning": ["Lightning"],
    "wildfire": ["Wildfire"],
}

FEMA = {
    1: ["CT", "ME", "MA", "NH", "RI", "VT"],
    2: ["NJ", "NY", "PR", "VI"],
    3: ["DE", "MD", "PA", "VA", "DC", "WV"],
    4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
    5: ["IL", "IN", "MI", "MN", "OH", "WI"],
    6: ["AR", "LA", "NM", "OK", "TX"],
    7: ["IA", "KS", "MO", "NE"],
    8: ["CO", "MT", "ND", "SD", "UT", "WY"],
    9: ["AZ", "CA", "HI", "NV", "GU", "AS", "MP"],
    10: ["AK", "ID", "OR", "WA"],
}
