"""
Upload data/processed/ to the team's private Hugging Face dataset.
Only one person needs to run this, after scripts/download_data.py.

    hf auth login                      # paste a WRITE token
    python scripts/upload_to_hf.py     # uses config.HF_REPO_ID
    python scripts/upload_to_hf.py --repo-id myorg/storm-outage-data

Re-running uploads only files that changed.
"""
import argparse
from importlib import import_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

CARD = """---
license: other
pretty_name: Storm-to-Outage (Southeast US)
---
# Storm-to-Outage data (Southeast US, 2014-2025)

Processed data for the GSU DSCI 4350 capstone. Built by `scripts/download_data.py`
in the project GitHub repo. Study area: AL, FL, GA, NC, SC, TN.

| Path | Contents | Source |
| --- | --- | --- |
| `eaglei/outages_YYYY.parquet` | Customers out per county every 15 min | EAGLE-I, Oak Ridge National Lab (figshare 24237376) |
| `eaglei/MCC.csv`, `coverage_history.csv`, `DQI.csv` | Customer counts, coverage, data quality | EAGLE-I |
| `storms/storm_events_YYYY.parquet` | NOAA Storm Events details | NOAA NCEI |
| `storms/storm_event_counties.parquet` | One row per storm event x county | derived |
| `reference/zone_county.parquet` | NWS forecast zone -> county | NWS |
| `reference/county_census.parquet` | Population, housing, land area | U.S. Census ACS 5-yr, Gazetteer |

Source data is U.S. government / national lab public data; cite the original sources.
"""


def main() -> None:
    try:
        HfApi = import_module("huggingface_hub").HfApi
    except ModuleNotFoundError:
        sys.exit("huggingface_hub is required; install it with: pip install huggingface_hub")

    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=C.HF_REPO_ID)
    p.add_argument("--public", action="store_true", help="make the dataset public")
    a = p.parse_args()

    if not C.PROCESSED_DIR.exists():
        sys.exit("data/processed/ not found - run scripts/download_data.py first")

    api = HfApi()
    print("Logged in as:", api.whoami()["name"])
    api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=a.repo_id, repo_type="dataset",
                    commit_message="Update dataset card")
    api.upload_folder(
        folder_path=str(C.PROCESSED_DIR),
        repo_id=a.repo_id,
        repo_type="dataset",
        allow_patterns=["**/*.parquet", "**/*.csv"],
        commit_message="Upload processed data",
    )
    print(f"Done: https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
