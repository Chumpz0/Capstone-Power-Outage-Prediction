r"""Combine the EAGLE-I ZIP into a CSV and copy its supporting files.

Run with: .venv/Scripts/python.exe pipeline/02_combine_eaglei.py
The combined CSV and its summary go to data/02_combined/. Supporting files
(DQI, MCC, coverage history, R scripts) go to data/reference/eaglei_archive/.
The source archive and existing Storm Events outputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from itertools import islice
from pathlib import Path, PurePosixPath

from paths import COMBINED, EAGLEI_ARCHIVE, RAW

YEARS = range(2014, 2026)
FIELDS = ["fips_code", "county", "state", "customers_out", "run_start_time"]
OUTPUT_FIELDS = ["SOURCE_YEAR", *FIELDS, "total_customers"]
OUTPUT_NAME = "eaglei_outages_2014_2025.csv"
SUMMARY_NAME = "eaglei_combination_summary.json"
CHUNK_ROWS = 100_000


def write_rows(output, digest, rows) -> None:
    """Serialize one bounded chunk, preserving all original field text."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    output.write(payload)
    digest.update(payload)


def combine(archive_path: Path, output_dir: Path, support_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate filenames.")
        # This archive is flat. Reject unsafe paths instead of extracting them.
        for name in names:
            if (
                PurePosixPath(name).name != name
                or "\\" in name
                or ":" in name
                or name in {".", ".."}
            ):
                raise ValueError(f"Unexpected archive path: {name!r}")

        yearly_names = [f"eaglei_outages_{year}.csv" for year in YEARS]
        missing = sorted(set(yearly_names) - set(names))
        if missing:
            raise ValueError(f"Missing requested years: {missing}")
        supporting_names = [name for name in names if name not in yearly_names]
        if {OUTPUT_NAME, SUMMARY_NAME} & set(supporting_names):
            raise ValueError("Archive filenames conflict with output filenames.")
        destinations = {OUTPUT_NAME: output_dir / OUTPUT_NAME, SUMMARY_NAME: output_dir / SUMMARY_NAME}
        destinations.update({name: support_dir / name for name in supporting_names})
        for destination in destinations.values():
            if destination.exists():
                raise FileExistsError(
                    f"Output already exists: {destination}. "
                    "Use --output-dir and --support-dir to choose separate new output folders."
                )

        # Validate every header before writing the large output.
        headers = {}
        for year, name in zip(YEARS, yearly_names):
            with archive.open(name) as member:
                header = next(csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig", newline="")))
            normalized = ["customers_out" if column == "sum" else column for column in header]
            if normalized not in (FIELDS, [*FIELDS, "total_customers"]):
                raise ValueError(f"Unsupported columns in {name}: {header}")
            headers[year] = header

        source_bytes = sum(info.file_size for info in members)
        # Covers the added SOURCE_YEAR column, blank total_customers, and margin.
        required_bytes = int(source_bytes * 1.5) + 1024**3
        free_bytes = shutil.disk_usage(output_dir).free
        if free_bytes < required_bytes:
            raise OSError(f"Need about {required_bytes / 1024**3:.1f} GiB free; available {free_bytes / 1024**3:.1f} GiB.")

        summary = {
            "archive": str(archive_path.resolve()),
            "output": OUTPUT_NAME,
            "columns": OUTPUT_FIELDS,
            "notes": [
                "Rows are appended by source year; within-file order is preserved.",
                "The 2023 sum column is renamed customers_out.",
                "total_customers is blank when not present in the source file.",
                "SOURCE_YEAR comes from the filename, not a timezone conversion.",
                "Original field text, leading zeroes, and missing values are preserved.",
            ],
            "years": [],
            "supporting_files": [],
        }
        started = time.monotonic()
        # Stage results so incomplete files do not appear under final filenames.
        with tempfile.TemporaryDirectory(prefix=".eaglei-build-", dir=output_dir) as temporary:
            staging = Path(temporary)
            digest = hashlib.sha256()
            total_rows = 0
            with (staging / OUTPUT_NAME).open("xb", buffering=4 * 1024**2) as output:
                write_rows(output, digest, [OUTPUT_FIELDS])
                for year, name in zip(YEARS, yearly_names):
                    header = headers[year]
                    width = len(header)
                    year_text = str(year)
                    row_count = 0
                    missing_counts = 0
                    other_years = 0
                    minimum_time = None
                    maximum_time = None
                    print(f"Combining {year}...", flush=True)
                    with archive.open(name) as member:
                        with io.TextIOWrapper(member, encoding="utf-8-sig", newline="") as text:
                            reader = csv.reader(text, strict=True)
                            next(reader)
                            while chunk := list(islice(reader, CHUNK_ROWS)):
                                for index, row in enumerate(chunk):
                                    if len(row) != width:
                                        raise ValueError(f"{name}, data row {row_count + index + 1}: expected {width} fields, found {len(row)}")
                                    timestamp = row[4]
                                    if timestamp:
                                        if minimum_time is None or timestamp < minimum_time:
                                            minimum_time = timestamp
                                        if maximum_time is None or timestamp > maximum_time:
                                            maximum_time = timestamp
                                    if not timestamp.startswith(year_text + "-"):
                                        other_years += 1
                                    if row[3] == "":
                                        missing_counts += 1
                                    if width == 5:
                                        row.append("")
                                    row.insert(0, year_text)
                                write_rows(output, digest, chunk)
                                row_count += len(chunk)
                                if row_count % 5_000_000 == 0:
                                    print(f"  {year}: {row_count:,} rows", flush=True)
                    total_rows += row_count
                    summary["years"].append({
                        "year": year,
                        "source_file": name,
                        "source_crc32": f"{archive.getinfo(name).CRC:08x}",
                        "source_header": header,
                        "rows": row_count,
                        "minimum_timestamp": minimum_time,
                        "maximum_timestamp": maximum_time,
                        "missing_customers_out": missing_counts,
                        "timestamps_outside_source_year_or_missing": other_years,
                    })
                    print(f"Finished {year}: {row_count:,} rows ({minimum_time} to {maximum_time})", flush=True)

            expected_hash = digest.hexdigest()
            print("Verifying the saved combined CSV against the written data...", flush=True)
            with (staging / OUTPUT_NAME).open("rb") as saved:
                actual_hash = hashlib.file_digest(saved, "sha256").hexdigest()
            if actual_hash != expected_hash:
                raise OSError("Combined CSV failed its SHA-256 verification.")

            # ZIP CRC checks are performed by zipfile as each source reaches EOF.
            for name in supporting_names:
                support_digest = hashlib.sha256()
                with archive.open(name) as source, (staging / name).open("xb") as destination:
                    while block := source.read(1024**2):
                        destination.write(block)
                        support_digest.update(block)
                with (staging / name).open("rb") as saved:
                    if hashlib.file_digest(saved, "sha256").hexdigest() != support_digest.hexdigest():
                        raise OSError(f"Copy verification failed: {name}")
                summary["supporting_files"].append({"name": name, "sha256": support_digest.hexdigest()})
                print(f"Copied {name}", flush=True)

            summary.update({
                "total_rows": total_rows,
                "output_bytes": (staging / OUTPUT_NAME).stat().st_size,
                "sha256": actual_hash,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            })
            (staging / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            for name, destination in destinations.items():
                (staging / name).replace(destination)
            print(f"Created {output_dir / OUTPUT_NAME}: {total_rows:,} rows.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RAW / "24237376.zip")
    parser.add_argument("--output-dir", type=Path, default=COMBINED)
    parser.add_argument("--support-dir", type=Path, default=EAGLEI_ARCHIVE)
    arguments = parser.parse_args()
    combine(arguments.archive, arguments.output_dir, arguments.support_dir)
