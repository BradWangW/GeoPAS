import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import cocopp
from cocopp import archiving, pproc


DEFAULT_ALGOS = [
    "BrentSTEPqi_Posik",
    "BrentSTEPrr_Posik",
    "CMA-CSA_Atamna",
    "HCMA_loshchilov_noiseless",
    "HMLSL_pal_noiseless",
    "IPOP400D_auger_noiseless",
    "MCS_huyer_noiseless",
    "MLSL_pal_noiseless",
    "OQNLP_pal_noiseless",
    "SMAC-BBOB_hutter_noiseless",
    "fmincon_pal_noiseless",
    "fminunc_pal_noiseless",
]


def _norm_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_archive_entries(
    archive: archiving.COCODataArchive, algos: list[str], allow_missing: bool
) -> tuple[list[str], list[str]]:
    """Return (archive_entries, missing_algorithms).

    Tries to query per-algorithm to avoid listing the entire archive.
    If an algorithm is not found, falls back to listing once and doing
    normalized name matching.
    """

    entries: list[str] = []
    missing: list[str] = []

    for algo in algos:
        try:
            matched = list(archive.get_all(algo))
        except Exception:
            matched = []
        if matched:
            entries.extend([str(p) for p in matched])
        else:
            missing.append(algo)

    if not missing:
        return _dedup_keep_order(entries), []

    # Fallback: list once, then do best-effort normalized matching
    all_entries = [str(p) for p in archive.get_all("")]
    by_norm_stem: dict[str, list[str]] = {}
    for p in all_entries:
        stem = Path(p).stem
        by_norm_stem.setdefault(_norm_name(stem), []).append(p)

    still_missing: list[str] = []
    for algo in missing:
        key = _norm_name(algo)
        direct = by_norm_stem.get(key)
        if direct:
            entries.extend(direct)
            continue

        # substring match (normalized)
        candidates = [
            p
            for p in all_entries
            if key in _norm_name(Path(p).stem)
        ]
        if len(candidates) == 1:
            entries.append(candidates[0])
        elif candidates:
            # take the shortest match (usually most specific)
            candidates = sorted(candidates, key=lambda s: len(Path(s).stem))
            entries.append(candidates[0])
            print(
                f"WARNING: multiple archive matches for '{algo}', using '{Path(candidates[0]).stem}'."
            )
        else:
            still_missing.append(algo)

    if still_missing and not allow_missing:
        raise ValueError(f"Algorithms not found in archive: {still_missing}")

    return _dedup_keep_order(entries), still_missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract per-(func, dim, instance) evaluation performance for selected BBOB algorithms "
            "from the official COCO archive, without running full cocopp postprocessing."
        )
    )
    ap.add_argument(
        "--archive",
        default="bbob",
        help=(
            "COCO archive name OR local archive directory. "
            "Examples: 'bbob' (official) or 'bbob-exp' (local dir created via archiving.create)."
        ),
    )
    ap.add_argument(
        "--create-archive",
        action="store_true",
        default=False,
        help=(
            "If --archive points to a local directory that does not exist, create/download it via "
            "cocopp.archiving.create(archive_dir)."
        ),
    )
    ap.add_argument(
        "--target",
        type=float,
        default=0.01,
        help="Target value passed to detEvals/_detMaxEvals (default: 0.01)",
    )
    ap.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10],
        help="Dimensions to include (default: 2 3 5 10 20)",
    )
    ap.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Instances of interest (default: 1 2 3 4 5)",
    )
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGOS,
        help="Algorithm names to include (default: the 12 requested algorithms)",
    )
    ap.add_argument(
        "--outdir",
        default="data",
        help="Output directory (default: current directory)",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        default=True,
        help="Skip algorithms not found in the archive (default: True)",
    )
    ap.add_argument(
        "--no-allow-missing",
        dest="allow_missing",
        action="store_false",
        help="Fail if any algorithm is missing from the archive",
    )

    args = ap.parse_args()

    # Respect proxy settings if user exports them (or if they were set in a notebook).
    # You can also set these before running: HTTP_PROXY / HTTPS_PROXY

    cocopp.testbedsettings.GECCOBBOBTestbed.settings["instancesOfInterest"] = list(
        args.instances
    )
    cocopp.config.config()

    # Support both official named archives (e.g. 'bbob') and local archive directories.
    archive_arg = str(args.archive)
    if Path(archive_arg).exists() and Path(archive_arg).is_dir():
        archive = archiving.get(archive_arg)
    else:
        if args.create_archive and ("/" in archive_arg or archive_arg.endswith("-exp")):
            # Heuristic: treat as local dir name when the user intends a local archive.
            # This mirrors ERT_cal.ipynb, which does archiving.create('bbob-exp').
            print(f"Creating local COCO archive at: {archive_arg}")
            archiving.create(archive_arg)
            archive = archiving.get(archive_arg)
        else:
            archive = archiving.get(archive_arg)

    archive_entries, missing = resolve_archive_entries(
        archive, list(args.algorithms), allow_missing=args.allow_missing
    )

    if missing:
        print(f"WARNING: algorithms not found and will be skipped: {missing}")

    print(f"Loading {len(archive_entries)} archive entries via pproc.processInputArgs...")
    all_datasets, pathnames, datasetlists_by_path = pproc.processInputArgs(archive_entries)

    records: list[dict] = []
    for path, dataset_list in datasetlists_by_path.items():
        algo_name = Path(path).stem
        for ds in dataset_list:
            if int(ds.dim) not in set(args.dims):
                continue

            max_evals = ds._detMaxEvals(args.target)
            hit_evals = ds.detEvals([args.target])[0]

            instances = ds.instancenumbers
            _, unique_indices = np.unique(instances, return_index=True)

            filtered_max = max_evals[unique_indices]
            filtered_hit = hit_evals[unique_indices]

            success = ~np.isnan(filtered_hit)
            evals_used = np.where(success, filtered_hit, filtered_max)

            for idx, pos in enumerate(unique_indices):
                records.append(
                    {
                        "algo": algo_name,
                        "funcId": int(ds.funcId),
                        "Dim": int(ds.dim),
                        "iid": int(instances[pos]),
                        "evals": float(evals_used[idx]),
                        "success": bool(success[idx]),
                    }
                )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    performance_records_path = outdir / "performance_records_10.csv"
    iid_perform_path = outdir / "iid_perform_10.csv"

    performance_df = pd.DataFrame(records)
    performance_df.to_csv(performance_records_path, index=False)

    # Explain missing dimensions early (common source of "Dim==40 only has one algo").
    requested_dims = set(int(d) for d in args.dims)
    if not performance_df.empty:
        dims_by_algo = (
            performance_df.groupby("algo")["Dim"].unique().to_dict()
        )
        missing_dims_by_algo: dict[str, list[int]] = {}
        for algo in sorted(set(args.algorithms)):
            present = set(int(d) for d in dims_by_algo.get(algo, []))
            missing_dims = sorted(requested_dims - present)
            if missing_dims:
                missing_dims_by_algo[algo] = missing_dims
        if missing_dims_by_algo:
            print("WARNING: some requested dimensions are missing in the COCO archive for these algorithms:")
            for algo, missing_dims in missing_dims_by_algo.items():
                print(f"  - {algo}: missing dims {missing_dims}")

    # Match the notebook's ERT-style decomposition: each run contributes evals / (#successes in that group)
    unique_instances = performance_df.drop_duplicates(subset=["algo", "funcId", "Dim", "iid"])
    group_success = (
        unique_instances.groupby(["algo", "funcId", "Dim"])["success"]
        .sum()
        .reset_index()
        .rename(columns={"success": "group_success_sum"})
    )
    merged_df = performance_df.merge(group_success, on=["algo", "funcId", "Dim"], how="left")

    merged_df["avg_evals"] = np.where(
        merged_df["group_success_sum"] > 0,
        merged_df["evals"] / merged_df["group_success_sum"],
        np.nan,
    )

    result_df = merged_df[["algo", "funcId", "Dim", "iid", "avg_evals"]]
    # Fill NaNs with the current max (same strategy as the notebook)
    fill_value = float(np.nanmax(result_df["avg_evals"].to_numpy()))
    if not np.isfinite(fill_value):
        raise ValueError(
            "No finite avg_evals values were computed. This usually means the selected archive entries "
            "contain no data for the requested dims/instances/target."
        )
    result_df = result_df.fillna(fill_value)

    df_wide = (
        result_df.pivot(index=["funcId", "Dim", "iid"], columns="algo", values="avg_evals")
        .reset_index()
        .sort_values(["funcId", "Dim", "iid"], kind="stable")
    )

    # If an algo has no rows for a (func, dim, iid) key, pivot introduces NaNs.
    # To mimic the notebook behavior ("missing" treated as worst-case), fill them too.
    df_wide = df_wide.fillna(fill_value)
    df_wide.to_csv(iid_perform_path, index=False)

    print(f"Wrote: {performance_records_path}")
    print(f"Wrote: {iid_perform_path}")


if __name__ == "__main__":
    main()
