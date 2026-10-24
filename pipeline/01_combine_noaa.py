"""Combine yearly NOAA Storm Events files into one CSV per section."""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

from paths import COMBINED as OUTPUT_DIR, RAW as INPUT_DIR

SECTIONS = ("details", "fatalities", "locations")
YEAR_PATTERN = re.compile(r"_d(\d{4})_")


def source_year(path: Path) -> int:
    """Extract the data year from a NOAA Storm Events filename."""
    match = YEAR_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not determine year from filename: {path.name}")
    return int(match.group(1))


def combine_section(section: str) -> tuple[Path, int, int]:
    """Stream all yearly files for one section into a single CSV."""
    input_files = sorted(
        INPUT_DIR.glob(f"StormEvents_{section}-ftp_v1.0_d*_c*.csv.gz"),
        key=source_year,
    )
    if not input_files:
        raise FileNotFoundError(f"No {section} files found in {INPUT_DIR}")

    output_path = OUTPUT_DIR / f"storm_events_{section}_all_years.csv"
    expected_header: list[str] | None = None
    total_rows = 0

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)

        for input_path in input_files:
            year = source_year(input_path)
            with gzip.open(
                input_path, "rt", encoding="utf-8-sig", newline=""
            ) as input_file:
                reader = csv.reader(input_file)
                header = next(reader, None)
                if header is None:
                    print(f"Skipping empty file: {input_path.name}")
                    continue

                if expected_header is None:
                    expected_header = header
                    writer.writerow(["SOURCE_YEAR", *header])
                elif header != expected_header:
                    raise ValueError(
                        f"Column mismatch in {input_path.name}. "
                        "The yearly files cannot be safely combined."
                    )

                rows_this_file = 0
                for row in reader:
                    writer.writerow([year, *row])
                    rows_this_file += 1
                total_rows += rows_this_file
                print(f"{section}: added {year} ({rows_this_file:,} rows)")

    return output_path, len(input_files), total_rows


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Reading yearly files from: {INPUT_DIR}")

    for section in SECTIONS:
        output_path, file_count, row_count = combine_section(section)
        print(
            f"Created {output_path.name} from {file_count} files "
            f"({row_count:,} data rows)."
        )


if __name__ == "__main__":
    main()
