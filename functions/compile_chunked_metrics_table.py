#!/usr/bin/env python3
"""Compile a table of [res, k_views, mean, median, p90] from a chunked CSV.

Expected input format (repeated chunks):

{parameters}_res_{res}_k_views_{k_views}
Mean
{table_mean}
Median
{table_median}
P90
{table_p90}

For each metric table, this script takes the bottom-right value (row label 'all',
column 'all'), i.e. the "all/all" entry.

Outputs a CSV with columns: res,k_views,mean,median,p90
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


HEADER_RE = re.compile(r"(?:^|_)res_(?P<res>\d+)_k_views_(?P<k_views>\d+)(?:$|\b|_)")


@dataclass
class ChunkResult:
    res: int
    k_views: int
    mean: float
    median: float
    p90: float


class ParseError(RuntimeError):
    pass


def _is_metric_label(line: str) -> bool:
    return line.strip() in {"Mean", "Median", "P90"}


def _looks_like_chunk_header(line: str) -> bool:
    if not line.strip():
        return False
    if _is_metric_label(line):
        return False
    # Guard against table header lines.
    if line.lstrip().startswith("Problem Group"):
        return False
    return HEADER_RE.search(line) is not None


def _parse_chunk_header(line: str) -> tuple[int, int]:
    m = HEADER_RE.search(line)
    if not m:
        raise ParseError(f"Could not find '_res_<n>_k_views_<n>' in header line: {line!r}")
    return int(m.group("res")), int(m.group("k_views"))


def _parse_metric_all_all(lines: list[str], start_idx: int, metric_name: str) -> tuple[float, int]:
    """Parse the metric table starting at start_idx, returning (value, next_idx).

    start_idx should point to the line immediately AFTER the metric label.
    """

    i = start_idx
    # Skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines) or not lines[i].lstrip().startswith("Problem Group"):
        raise ParseError(
            f"Expected a table header starting with 'Problem Group' after {metric_name}, got: "
            f"{lines[i].rstrip() if i < len(lines) else '<EOF>'}"
        )

    # Parse CSV rows until we hit a blank line or next metric or next chunk.
    all_row: Optional[list[str]] = None

    i += 1  # move past table header
    while i < len(lines):
        line = lines[i].strip("\n")
        stripped = line.strip()

        if not stripped:
            break
        if _is_metric_label(stripped):
            break
        if _looks_like_chunk_header(stripped):
            break

        # Parse this row as CSV.
        row = next(csv.reader([line]))
        if not row:
            i += 1
            continue

        row_label = row[0].strip()
        if row_label.lower() == "all":
            all_row = row

        i += 1

    if all_row is None:
        raise ParseError(f"Did not find 'all' row in {metric_name} table")

    if len(all_row) < 2:
        raise ParseError(f"Malformed 'all' row in {metric_name} table: {all_row!r}")

    value_str = all_row[-1].strip()
    try:
        value = float(value_str)
    except ValueError as e:
        raise ParseError(f"Could not parse all/all value {value_str!r} in {metric_name}") from e

    return value, i


def parse_chunked_metrics_file(path: Path) -> list[ChunkResult]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    results: list[ChunkResult] = []

    i = 0
    while i < len(lines):
        # Skip blanks
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        line = lines[i].strip("\n")

        # Find next chunk header
        if not _looks_like_chunk_header(line):
            i += 1
            continue

        res, k_views = _parse_chunk_header(line)
        i += 1

        # Expect Mean/Median/P90 in order (with optional blank lines between)
        def expect_metric_label(expected: str) -> None:
            nonlocal i
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines) or lines[i].strip() != expected:
                got = lines[i].strip() if i < len(lines) else "<EOF>"
                raise ParseError(f"Expected metric label {expected!r}, got {got!r} (res={res}, k_views={k_views})")
            i += 1

        expect_metric_label("Mean")
        mean, i = _parse_metric_all_all(lines, i, "Mean")

        expect_metric_label("Median")
        median, i = _parse_metric_all_all(lines, i, "Median")

        expect_metric_label("P90")
        p90, i = _parse_metric_all_all(lines, i, "P90")

        results.append(ChunkResult(res=res, k_views=k_views, mean=mean, median=median, p90=p90))

        # Move forward until next non-empty line (next chunk)
        while i < len(lines) and not lines[i].strip():
            i += 1

    if not results:
        raise ParseError(
            "No chunks parsed. Ensure headers contain '_res_<n>_k_views_<n>' and tables follow Mean/Median/P90 blocks."
        )

    return results


def write_compiled_csv(results: list[ChunkResult], out_f) -> None:
    writer = csv.writer(out_f)
    writer.writerow(["res", "k_views", "mean", "median", "p90"])
    for r in results:
        writer.writerow([r.res, r.k_views, r.mean, r.median, r.p90])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a chunked metrics CSV and compile rows with res/k_views and all/all values for Mean/Median/P90."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the chunked CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, prints to stdout.",
    )

    args = parser.parse_args(argv)

    try:
        results = parse_chunked_metrics_file(args.input_csv)
    except ParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.output is None:
        write_compiled_csv(results, sys.stdout)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            write_compiled_csv(results, f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
