"""Generate relERT tables from the official COCO/BBOB archive.

This script computes relERT using COCO's detERT definition at a given target.
It writes CSVs compatible with this repo's training/eval pipeline:
  data/relert_bbob_{dim}.csv

By default, it uses instancesOfInterest=[1..5] and target=0.01.

Note: The repo historically uses the column name 'M_LSL_pal_noiseless' for the
MLSL algorithm. We keep that alias for backward compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _problem_name(func_id: int) -> str:
    return f"f{int(func_id)}"


def _load_datasets(
    *, archive_name: str, algos: list[str], instances: list[int]
) -> dict[str, list]:
    cocopp.testbedsettings.GECCOBBOBTestbed.settings["instancesOfInterest"] = list(
        instances
    )
    cocopp.config.config()

    archive = archiving.get(archive_name)

    entries: list[str] = []
    for algo in algos:
        matched = list(archive.get_all(algo))
        if not matched:
            raise ValueError(
                f"Algorithm '{algo}' not found in COCO archive '{archive_name}'."
            )
        entries.extend([str(p) for p in matched])

    _, _, datasetlists_by_path = pproc.processInputArgs(entries)

    out: dict[str, list] = {}
    for path, dlist in datasetlists_by_path.items():
        algo_name = Path(path).stem
        out[algo_name] = dlist
    return out


def build_relert_table(
    *,
    datasets_by_algo: dict[str, list],
    algos: list[str],
    dims: list[int],
    target: float,
    penalty: float,
) -> pd.DataFrame:
    # Collect ERT per (Problem, Dim) per algo.
    rows: dict[tuple[str, int], dict[str, float]] = {}

    dims_set = set(int(d) for d in dims)

    for algo, dlist in datasets_by_algo.items():
        for ds in dlist:
            if int(ds.dim) not in dims_set:
                continue
            key = (_problem_name(int(ds.funcId)), int(ds.dim))
            rows.setdefault(key, {})
            # COCO detERT returns an array aligned to input targets.
            ert_val = float(ds.detERT([target])[0])
            rows[key][algo] = ert_val

    if not rows:
        raise ValueError(f"No datasets found for dims={dims}.")

    index = pd.MultiIndex.from_tuples(sorted(rows.keys()), names=["Problem", "Dim"])

    # Always materialize the requested algorithm columns, even if some are missing
    # for a given dimension in the archive.
    ert_df = pd.DataFrame(index=index, columns=list(algos), dtype=float)
    for key, vals in rows.items():
        for algo, ert in vals.items():
            ert_df.loc[key, algo] = ert

    # relERT = ERT / best ERT per row
    best = ert_df.min(axis=1, skipna=True)
    relert = ert_df.div(best, axis=0)

    # Fill non-finite / missing with penalty (matches existing pipeline)
    relert = relert.replace([np.inf, -np.inf], np.nan).fillna(float(penalty))

    # Stable ordering: for each problem f1..f24, list dims in `dims` order.
    ordered_index = []
    for f in range(1, 25):
        prob = _problem_name(f)
        for d in dims:
            ordered_index.append((prob, int(d)))
    relert = relert.loc[pd.MultiIndex.from_tuples(ordered_index, names=["Problem", "Dim"])]

    return relert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive",
        default="bbob",
        help="COCO archive name (default: bbob)",
    )
    ap.add_argument(
        "--max-dims",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help=(
            "Generate relERT tables up to each max dimension. "
            "Example: 10 => dims [2,3,5,10], 20 => [2,3,5,10,20], 40 => [2,3,5,10,20,40]. "
            "Default: 10 20 40"
        ),
    )
    ap.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Instances of interest (default: 1 2 3 4 5)",
    )
    ap.add_argument(
        "--target",
        type=float,
        default=0.01,
        help="Target passed to detERT (default: 0.01)",
    )
    ap.add_argument(
        "--penalty",
        type=float,
        default=36690.3,
        help="Penalty value to fill missing relERT (default: 36690.3)",
    )
    ap.add_argument(
        "--outdir",
        default="data",
        help="Output directory (default: data)",
    )
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGOS,
        help="Algorithms to include (default: the 12 used in this repo)",
    )
    ap.add_argument(
        "--alias-mlsl",
        action="store_true",
        default=True,
        help="Write MLSL column as 'M_LSL_pal_noiseless' for backward compatibility (default: true)",
    )
    ap.add_argument(
        "--no-alias-mlsl",
        dest="alias_mlsl",
        action="store_false",
        help="Keep official COCO name 'MLSL_pal_noiseless'",
    )

    args = ap.parse_args()

    algos = list(args.algorithms)

    datasets_by_algo = _load_datasets(
        archive_name=args.archive, algos=algos, instances=list(args.instances)
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build and write per-dim tables.
    bbob_dims = [2, 3, 5, 10, 20, 40]

    for max_dim in args.max_dims:
        dims = [d for d in bbob_dims if d <= int(max_dim)]
        relert = build_relert_table(
            datasets_by_algo=datasets_by_algo,
            algos=algos,
            dims=dims,
            target=float(args.target),
            penalty=float(args.penalty),
        )

        if args.alias_mlsl and "MLSL_pal_noiseless" in relert.columns:
            relert = relert.rename(columns={"MLSL_pal_noiseless": "M_LSL_pal_noiseless"})

        # Keep a stable column order (match existing relert_bbob_*.csv layout)
        desired_order = [
            "BrentSTEPqi_Posik",
            "BrentSTEPrr_Posik",
            "CMA-CSA_Atamna",
            "HCMA_loshchilov_noiseless",
            "HMLSL_pal_noiseless",
            "IPOP400D_auger_noiseless",
            "MCS_huyer_noiseless",
            "M_LSL_pal_noiseless" if args.alias_mlsl else "MLSL_pal_noiseless",
            "OQNLP_pal_noiseless",
            "SMAC-BBOB_hutter_noiseless",
            "fmincon_pal_noiseless",
            "fminunc_pal_noiseless",
        ]
        relert = relert[desired_order]

        out_path = outdir / f"relert_bbob_{int(max_dim)}.csv"
        relert.to_csv(out_path)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
