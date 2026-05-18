#!/usr/bin/env python3
"""Parallel CV runner that schedules tasks across GPUs.

For multi-protocol runs, outputs are written under per-protocol subdirectories.
For single-protocol runs, outputs are written directly into ``--out-dir``.

Supported protocols: random, lpo, lio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_int_list(s: str) -> List[int]:
    if s.strip() == "":
        return []
    parts = re.split(r"[ ,]+", s.strip())
    out: List[int] = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out


def _detect_gpus_via_nvidia_smi() -> List[int]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except Exception:
        return []
    gpus: List[int] = []
    for line in out.splitlines():
        m = re.match(r"^GPU\s+(\d+):", line.strip())
        if m:
            gpus.append(int(m.group(1)))
    return gpus


def _resolve_gpus(gpus_arg: str) -> List[int]:
    if gpus_arg == "auto":
        gpus = _detect_gpus_via_nvidia_smi()
        if gpus:
            return gpus
        # fallback: respect CUDA_VISIBLE_DEVICES if present
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            # CUDA_VISIBLE_DEVICES can be UUIDs; we only support numeric lists here.
            try:
                return [int(x) for x in cvd.split(",") if x.strip() != ""]
            except ValueError:
                return [0]
        return [0]

    return _parse_int_list(gpus_arg)


@dataclass(frozen=True)
class Task:
    protocol: str
    task_id: str
    payload: Dict[str, Any]


def _write_metrics_csv(metrics: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _as_frame(value: object) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(value).__name__}")
        return value

    def _as_series(value: object) -> pd.Series:
        if not isinstance(value, pd.Series):
            raise TypeError(f"Expected Series, got {type(value).__name__}")
        return value

    summary_blocks = [
        ("Mean", "scores", "vbs", "sbs", "gap_closure"),
        ("Median", "median_scores", "median_vbs", "median_sbs", "median_gap_closure"),
        ("P90", "p90_scores", "p90_vbs", "p90_sbs", "p90_gap_closure"),
    ]

    with out_path.open("w") as f:
        def _write_block(title: str, *, scores_key: str, vbs_key: str, sbs_key: str, gap_key: str) -> None:
            f.write(f"{title}\n")
            f.write("AS\n")
            _as_frame(metrics[scores_key]).to_csv(f, index=False)
            f.write("\nVBS\n")
            _as_frame(metrics[vbs_key]).to_csv(f, index=False)
            f.write("\nSBS\n")
            _as_frame(metrics[sbs_key]).to_csv(f, index=False)
            f.write("\nGap_Closure\n")
            _as_frame(metrics[gap_key]).to_csv(f, index=False)

        def _write_frame_section(title: str, key: str, *, optional: bool = False) -> None:
            value = metrics.get(key)
            if value is None:
                if optional:
                    return
                raise KeyError(key)
            f.write(f"\n{title}\n")
            _as_frame(value).to_csv(f, index=False)

        def _write_series_section(title: str, key: str) -> None:
            f.write(f"\n{title}\n")
            _as_series(metrics[key]).to_csv(f, header=["rate"])

        for idx, (title, scores_key, vbs_key, sbs_key, gap_key) in enumerate(summary_blocks):
            if idx:
                f.write("\n")
            _write_block(title, scores_key=scores_key, vbs_key=vbs_key, sbs_key=sbs_key, gap_key=gap_key)

        _write_frame_section("Accuracies", "accuracies")
        _write_frame_section("F1", "f1")
        _write_series_section("Pick_Rate", "pick_rate")
        _write_series_section("VBS_Pick_Rate", "vbs_pick_rate")


def _canonical_problem_list(df: pd.DataFrame) -> List[str]:
    # Canonicalize to 'f{fid}'
    probs = sorted(pd.unique(df["Problem"].astype(str).str.lower()))
    out: List[str] = []
    for p in probs:
        m = re.match(r"^[fF]?(\d+)$", p)
        if not m:
            continue
        out.append(f"f{int(m.group(1))}")
    # De-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _sanitize_tb_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9._\-/]+", "", s)
    s = s.strip("._/")
    return s or "run"


def _log_fold_average_curves(
    *,
    tb_log_dir: Optional[str],
    protocol: str,
    preds_by_fold: Dict[str, pd.DataFrame],
    run_stub: str,
) -> None:
    if not tb_log_dir:
        return

    histories: List[Dict[str, object]] = []
    for fold_id in sorted(preds_by_fold.keys()):
        payload = getattr(preds_by_fold[fold_id], "attrs", {}).get("tb_history")
        if isinstance(payload, dict):
            histories.append(payload)

    if not histories:
        print(f"[TensorBoard] No per-fold histories found for protocol={protocol}; skip fold-average logging.")
        return

    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except Exception as e:
        print(f"[TensorBoard] SummaryWriter unavailable; skip fold-average logging ({e}).")
        return

    metric_names = [
        "train/loss",
        "train/loss_main",
        "train/loss_head_2",
        "train/as",
        "train/lr",
        "val/loss",
        "val/loss_main",
        "val/loss_head_2",
        "val/as",
    ]
    run_name = _sanitize_tb_name(f"{protocol}/{run_stub}/fold_average")
    event_dir = Path(tb_log_dir) / run_name
    event_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(event_dir))
    try:
        for metric in metric_names:
            max_len = 0
            series_all: List[np.ndarray] = []
            for h in histories:
                vals = h.get(metric, []) if isinstance(h, dict) else []
                arr = np.asarray(vals, dtype=np.float64)
                series_all.append(arr)
                if arr.size > max_len:
                    max_len = int(arr.size)

            if max_len == 0:
                continue

            mat = np.full((len(series_all), max_len), np.nan, dtype=np.float64)
            for i, arr in enumerate(series_all):
                if arr.size > 0:
                    mat[i, : arr.size] = arr

            for epoch in range(max_len):
                vals = mat[:, epoch]
                ok = np.isfinite(vals)
                if not np.any(ok):
                    continue
                mean_val = float(np.mean(vals[ok]))
                writer.add_scalar(f"{metric}_mean_over_folds", mean_val, epoch)
                writer.add_scalar(f"{metric}_n_folds", int(np.sum(ok)), epoch)
    finally:
        writer.flush()
        writer.close()

    print(f"[TensorBoard] Wrote fold-average curves for protocol={protocol} to {event_dir}")


def _make_tasks(
    *,
    protocol: str,
    df: pd.DataFrame,
    instances_all: Sequence[int],
    n_splits: int,
) -> List[Task]:
    instances_all = [int(i) for i in instances_all]

    if protocol == "lio":
        return [
            Task(protocol="lio", task_id=f"inst_{inst}", payload={"test_inst": int(inst)})
            for inst in instances_all
        ]

    if protocol == "lpo":
        probs = _canonical_problem_list(df)
        present = set(df["Problem"].astype(str).str.lower())
        probs = [p for p in probs if p in present]
        return [
            Task(protocol="lpo", task_id=f"prob_{p}", payload={"test_prob": p, "fold_idx": i})
            for i, p in enumerate(probs)
        ]

    if protocol == "random":
        return [
            Task(protocol="random", task_id=f"split_{i}", payload={"split_idx": i})
            for i in range(int(n_splits))
        ]

    raise ValueError(f"Unknown protocol: {protocol}")


def _result_file_suffix(args: argparse.Namespace) -> str:
    head_2_target_scale = str(args.head_2_target_scale or args.target_scale)
    return (
        f"priorscale_{args.prior_scale}"
        f"_sigmoidlogs_{args.sigmoid_log_s}"
        f"_tailscale_{args.tail_scale}"
        f"_head2lossweight_{args.head_2_loss_weight}"
        f"_head2scoreweight_{args.head_2_score_weight}"
        f"_head2targetscale_{head_2_target_scale}"
        f"_lamprior_{args.lam_prior}"
        f"_res_{args.resolution}"
        f"_k_views_{args.k_views}"
    )


def _protocols_from_arg(protocol_arg: str) -> List[str]:
    return ["random", "lpo", "lio"] if protocol_arg == "all" else [protocol_arg]


def _protocol_results_dir_name(protocol: str) -> str:
    return {"lpo": "LPO", "lio": "LIO", "random": "RANDOM"}.get(protocol, protocol.upper())


def _protocol_output_dir(*, out_dir: Path, protocol: str, protocol_count: int) -> Path:
    return out_dir if protocol_count == 1 else (out_dir / _protocol_results_dir_name(protocol))


def _protocol_results_dir_candidates(*, out_dir: Path, protocol: str) -> List[Path]:
    candidates = [out_dir, out_dir / _protocol_results_dir_name(protocol)]
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _write_protocol_outputs(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    protocols: Sequence[str],
    preds_by_protocol: Dict[str, Dict[str, pd.DataFrame]],
    metrics_by_protocol: Optional[Dict[str, Dict[str, object]]] = None,
    args: argparse.Namespace,
    tb_log_dir: Optional[str],
) -> None:
    from functions.model_interface import output_results

    print("Resolution", args.resolution, "K-Views", args.k_views)
    for p in protocols:
        preds_by_fold = preds_by_protocol.get(p, {})
        if not preds_by_fold:
            continue

        metrics = None if metrics_by_protocol is None else metrics_by_protocol.get(p)
        if metrics is None:
            metrics = output_results(df, preds_by_fold, protocol=p)
        results_dir = _protocol_output_dir(out_dir=out_dir, protocol=p, protocol_count=len(protocols))
        suffix = _result_file_suffix(args)
        out_csv = results_dir / f"res_{suffix}.csv"
        _write_metrics_csv(metrics, out_csv)
        print(f"Wrote {p} results CSV: {out_csv}")

        df_preds_all = pd.concat(list(preds_by_fold.values()), ignore_index=True)
        preds_out = results_dir / f"preds_{suffix}.csv.gz"
        df_preds_all.to_csv(preds_out, index=False, compression="gzip")
        print(f"Wrote {p} per-sample predictions: {preds_out}")

        _log_fold_average_curves(
            tb_log_dir=tb_log_dir,
            protocol=p,
            preds_by_fold=preds_by_fold,
            run_stub=out_dir.name,
        )


def _average_history_dicts(histories: Sequence[object]) -> Optional[Dict[str, List[float]]]:
    valid_histories = [h for h in histories if isinstance(h, dict)]
    if not valid_histories:
        return None

    metric_names = sorted({key for hist in valid_histories for key in hist.keys()})
    averaged: Dict[str, List[float]] = {}
    for metric in metric_names:
        arrays: List[np.ndarray] = []
        max_len = 0
        for hist in valid_histories:
            values = hist.get(metric, [])
            try:
                arr = np.asarray(values, dtype=float)
            except Exception:
                arr = np.asarray([], dtype=float)
            arrays.append(arr)
            max_len = max(max_len, int(arr.size))

        if max_len == 0:
            continue

        mat = np.full((len(arrays), max_len), np.nan, dtype=float)
        for i, arr in enumerate(arrays):
            if arr.size > 0:
                mat[i, : arr.size] = arr

        series: List[float] = []
        for epoch in range(max_len):
            vals = mat[:, epoch]
            ok = np.isfinite(vals)
            if not np.any(ok):
                continue
            series.append(float(np.mean(vals[ok])))
        if series:
            averaged[metric] = series

    return averaged or None


def _average_seed_fold_predictions(
    frames: Sequence[pd.DataFrame],
    *,
    alg_cols: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("Cannot average an empty list of prediction frames")

    key_cols = [col for col in ("Problem", "Dim", "Instance", "Repetition") if col in frames[0].columns]
    if not key_cols:
        raise ValueError("Prediction frame is missing key columns needed for seed aggregation")

    base = frames[0].copy().set_index(key_cols).sort_index()
    non_alg_cols = [col for col in base.columns if col not in alg_cols]
    alg_arrays: List[np.ndarray] = []

    for frame in frames:
        aligned = frame.copy().set_index(key_cols).sort_index()
        if list(aligned.columns) != list(base.columns):
            raise ValueError("Prediction frame columns differ across seeds")
        if not base.index.equals(aligned.index):
            raise ValueError("Prediction frame row keys differ across seeds")
        if non_alg_cols and not base[non_alg_cols].equals(aligned[non_alg_cols]):
            raise ValueError("Prediction frame metadata differs across seeds")
        alg_arrays.append(aligned.loc[:, list(alg_cols)].to_numpy(dtype=float))

    out = base.copy()
    out.loc[:, list(alg_cols)] = np.mean(np.stack(alg_arrays, axis=0), axis=0)
    out = out.reset_index()

    attrs = dict(frames[0].attrs)
    averaged_history = _average_history_dicts([getattr(frame, "attrs", {}).get("tb_history") for frame in frames])
    if averaged_history is not None:
        attrs["tb_history"] = averaged_history
    out.attrs = attrs
    return out


def _concat_predictions_by_fold(preds_by_fold: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not preds_by_fold:
        raise ValueError("Cannot concatenate empty fold predictions")
    return pd.concat([preds_by_fold[fold_id] for fold_id in sorted(preds_by_fold)], ignore_index=True)


def _normalize_problem_id(problem: object) -> str:
    text = str(problem).strip().lower()
    match = re.match(r"^[fF]?(\d+)$", text)
    return f"f{int(match.group(1))}" if match else text


def _rebuild_fold_predictions_from_concat(
    *,
    preds_all: pd.DataFrame,
    protocol: str,
) -> Dict[str, pd.DataFrame]:
    if preds_all.empty:
        return {}

    if protocol == "lpo":
        if "Problem" not in preds_all.columns:
            raise ValueError("Legacy LPO predictions are missing the Problem column")
        out: Dict[str, pd.DataFrame] = {}
        problem_ids = preds_all["Problem"].map(_normalize_problem_id)
        for problem in sorted(problem_ids.unique()):
            out[f"prob_{problem}"] = preds_all.loc[problem_ids == problem].reset_index(drop=True)
        return out

    if protocol == "lio":
        if "Instance" not in preds_all.columns:
            raise ValueError("Legacy LIO predictions are missing the Instance column")
        out = {}
        instances = preds_all["Instance"].astype(int)
        for inst in sorted(instances.unique()):
            out[f"inst_{int(inst)}"] = preds_all.loc[instances == int(inst)].reset_index(drop=True)
        return out

    return {"split_0": preds_all.reset_index(drop=True)}


def _load_legacy_fold_predictions(
    *,
    out_dir: Path,
    protocol: str,
    result_suffix: str,
) -> Dict[str, pd.DataFrame]:
    for results_dir in _protocol_results_dir_candidates(out_dir=out_dir, protocol=protocol):
        preds_path = results_dir / f"preds_{result_suffix}.csv.gz"
        if not preds_path.is_file():
            continue

        preds_all = pd.read_csv(preds_path)
        rebuilt = _rebuild_fold_predictions_from_concat(preds_all=preds_all, protocol=protocol)
        if rebuilt:
            note = "using single reconstructed split for random" if protocol == "random" else "reconstructed fold groups"
            print(f"[aggregate] Recovered {protocol} predictions from CSV: {preds_path} ({note})")
        return rebuilt

    return {}


def _load_protocol_predictions(
    *,
    out_dir: Path,
    protocol: str,
    result_suffix: str,
) -> Dict[str, pd.DataFrame]:
    protocol_dir = _fold_preds_root(out_dir) / protocol
    if protocol_dir.is_dir():
        preds_by_fold = {fp.stem: pd.read_pickle(fp) for fp in sorted(protocol_dir.glob("*.pkl"))}
        if preds_by_fold:
            return preds_by_fold
    return _load_legacy_fold_predictions(
        out_dir=out_dir,
        protocol=protocol,
        result_suffix=result_suffix,
    )


def _aggregate_seed_predictions(
    *,
    df: pd.DataFrame,
    seed_outputs: Sequence[Dict[str, Dict[str, pd.DataFrame]]],
    protocols: Sequence[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if not seed_outputs:
        return {}

    alg_cols = list(df.columns[2:])
    aggregated: Dict[str, Dict[str, pd.DataFrame]] = {}
    for protocol in protocols:
        def _aggregate_via_concat(*, reason: str) -> Dict[str, pd.DataFrame]:
            averaged_preds_all = _average_seed_fold_predictions(
                [_concat_predictions_by_fold(run) for run in nonempty_runs],
                alg_cols=alg_cols,
            )
            rebuilt = _rebuild_fold_predictions_from_concat(
                preds_all=averaged_preds_all,
                protocol=protocol,
            )
            if protocol == "random":
                print(
                    "[aggregate] Random folds are not seed-aligned "
                    f"({reason}); falling back to row-level averaging with a single reconstructed split"
                )
            else:
                print(
                    f"[aggregate] Fold groups are not seed-aligned for protocol={protocol} "
                    f"({reason}); reconstructed fold groups from concatenated predictions"
                )
            return rebuilt

        protocol_runs = [seed_output.get(protocol, {}) for seed_output in seed_outputs]
        nonempty_runs = [run for run in protocol_runs if run]
        if not nonempty_runs:
            aggregated[protocol] = {}
            continue

        if len(nonempty_runs) != len(protocol_runs):
            raise ValueError(
                f"Missing saved predictions for protocol={protocol} in one or more seed directories; "
                "re-run those seed outputs or ensure matching legacy preds_*.csv.gz files exist"
            )

        fold_sets = [set(run.keys()) for run in nonempty_runs]
        if any(fold_set != fold_sets[0] for fold_set in fold_sets[1:]):
            aggregated[protocol] = _aggregate_via_concat(reason="fold ids differ across seeds")
            continue

        aggregated_by_fold: Dict[str, pd.DataFrame] = {}
        try:
            for fold_id in sorted(fold_sets[0]):
                aggregated_by_fold[fold_id] = _average_seed_fold_predictions(
                    [run[fold_id] for run in nonempty_runs],
                    alg_cols=alg_cols,
                )
        except ValueError as exc:
            aggregated[protocol] = _aggregate_via_concat(reason=str(exc))
            continue

        aggregated[protocol] = aggregated_by_fold

    return aggregated


def _nanmean_preserve_all_nan(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(values), axis=axis)
    summed = np.nansum(values, axis=axis)
    mean = np.full(summed.shape, np.nan, dtype=float)
    np.divide(summed, counts, out=mean, where=counts > 0)
    return mean


def _average_seed_metric_frame(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("Cannot average an empty list of metric tables")

    key_col = str(frames[0].columns[0])
    base = frames[0].copy().set_index(key_col)
    arrays: List[np.ndarray] = []
    for frame in frames:
        aligned = frame.copy().set_index(key_col)
        if list(aligned.columns) != list(base.columns):
            raise ValueError("Metric table columns differ across seeds")
        if not base.index.equals(aligned.index):
            raise ValueError("Metric table row labels differ across seeds")
        arrays.append(aligned.to_numpy(dtype=float))

    mean = _nanmean_preserve_all_nan(np.stack(arrays, axis=0), axis=0)
    out = pd.DataFrame(mean, index=base.index, columns=base.columns).reset_index()
    out.columns = pd.Index([key_col] + list(base.columns), name=frames[0].columns.name)
    return out.round(3)


def _average_seed_metric_series(series_list: Sequence[pd.Series]) -> pd.Series:
    if not series_list:
        raise ValueError("Cannot average an empty list of metric series")

    union_index = pd.Index(sorted({idx for series in series_list for idx in series.index.astype(str)}), dtype=object)
    aligned = [series.astype(float).rename(index=str).reindex(union_index, fill_value=0.0) for series in series_list]
    mean = pd.concat(aligned, axis=1).mean(axis=1)
    mean = mean.sort_values(ascending=False)
    mean.name = series_list[0].name
    mean.index.name = series_list[0].index.name
    return mean


def _aggregate_seed_metrics(
    *,
    df: pd.DataFrame,
    seed_outputs: Sequence[Dict[str, Dict[str, pd.DataFrame]]],
    protocols: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    from functions.model_interface import output_results

    aggregated: Dict[str, Dict[str, object]] = {}
    for protocol in protocols:
        protocol_runs = [seed_output.get(protocol, {}) for seed_output in seed_outputs]
        nonempty_runs = [run for run in protocol_runs if run]
        if not nonempty_runs:
            aggregated[protocol] = {}
            continue

        if len(nonempty_runs) != len(protocol_runs):
            raise ValueError(
                f"Missing saved predictions for protocol={protocol} in one or more seed directories; "
                "re-run those seed outputs or ensure matching legacy preds_*.csv.gz files exist"
            )

        seed_metrics = [output_results(df, run, protocol=protocol, print_fold_summary=False) for run in nonempty_runs]
        template = seed_metrics[0]
        averaged_metrics: Dict[str, object] = {}
        for key, value in template.items():
            if key == "preds_all" or key.endswith("_by_fold"):
                continue

            values = [metrics.get(key) for metrics in seed_metrics]
            present = [item for item in values if item is not None]
            if not present:
                averaged_metrics[key] = None
                continue
            if len(present) != len(values):
                raise ValueError(f"Metric {key} is missing for one or more seeds in protocol={protocol}")

            sample = present[0]
            if isinstance(sample, pd.DataFrame):
                averaged_metrics[key] = _average_seed_metric_frame(present)
            elif isinstance(sample, pd.Series):
                averaged_metrics[key] = _average_seed_metric_series(present)

        scores_table = averaged_metrics.get("scores")
        median_table = averaged_metrics.get("median_scores")
        p90_table = averaged_metrics.get("p90_scores")
        if not isinstance(scores_table, pd.DataFrame):
            raise TypeError("Expected DataFrame for aggregated metric scores")
        if not isinstance(median_table, pd.DataFrame):
            raise TypeError("Expected DataFrame for aggregated metric median_scores")
        if not isinstance(p90_table, pd.DataFrame):
            raise TypeError("Expected DataFrame for aggregated metric p90_scores")

        mean_as = float(scores_table.set_index("Problem Group").loc["all", "all"])
        median_as = float(median_table.set_index("Problem Group").loc["all", "all"])
        p90_as = float(p90_table.set_index("Problem Group").loc["all", "all"])
        print(
            f"[aggregate] protocol={protocol}, "
            f"AS mean: {mean_as:.3f}, AS median: {median_as:.3f}, AS P90: {p90_as:.3f}"
        )

        aggregated[protocol] = averaged_metrics

    return aggregated


def _fold_preds_root(out_dir: Path) -> Path:
    return out_dir / "_fold_preds"


def _model_scores_root(out_dir: Path) -> Path:
    return out_dir / "_model_scores"


def _save_fold_predictions(
    *,
    out_dir: Path,
    preds_by_protocol: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    root = _fold_preds_root(out_dir)
    for protocol, preds_by_fold in preds_by_protocol.items():
        protocol_dir = root / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        for stale in protocol_dir.glob("*.pkl"):
            stale.unlink()
        for fold_id, preds in preds_by_fold.items():
            preds.to_pickle(protocol_dir / f"{fold_id}.pkl")


def _save_model_scores(
    *,
    out_dir: Path,
    model_scores_by_protocol: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    root = _model_scores_root(out_dir)
    for protocol, scores_by_fold in model_scores_by_protocol.items():
        protocol_dir = root / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        for stale in protocol_dir.glob("*.pkl"):
            stale.unlink()
        for fold_id, scores in scores_by_fold.items():
            scores.to_pickle(protocol_dir / f"{fold_id}.pkl")


def _load_model_scores(
    *,
    out_dir: Path,
    protocols: Sequence[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    scores_by_protocol: Dict[str, Dict[str, pd.DataFrame]] = {}
    root = _model_scores_root(out_dir)
    for protocol in protocols:
        protocol_dir = root / protocol
        if not protocol_dir.is_dir():
            continue
        scores_by_fold = {fp.stem: pd.read_pickle(fp) for fp in sorted(protocol_dir.glob("*.pkl"))}
        if scores_by_fold:
            scores_by_protocol[protocol] = scores_by_fold
    return scores_by_protocol


def _load_fold_predictions(
    *,
    out_dir: Path,
    protocols: Sequence[str],
    result_suffix: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    preds_by_protocol: Dict[str, Dict[str, pd.DataFrame]] = {}
    for protocol in protocols:
        preds_by_fold = _load_protocol_predictions(
            out_dir=out_dir,
            protocol=protocol,
            result_suffix=result_suffix,
        )
        if preds_by_fold:
            preds_by_protocol[protocol] = preds_by_fold
    return preds_by_protocol


def _normalize_relert_df(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    if "Problem" in df_norm.columns:
        df_norm["Problem"] = df_norm["Problem"].astype(str).str.lower()
    if "Dim" in df_norm.columns:
        df_norm["Dim"] = df_norm["Dim"].astype(int)
    return df_norm


def _prior_source_df_for_fold(
    *,
    df: pd.DataFrame,
    protocol: str,
    fold_id: str,
    base_scores: pd.DataFrame,
) -> pd.DataFrame:
    df_norm = _normalize_relert_df(df)

    if protocol == "lpo":
        test_problems = [str(problem).lower() for problem in getattr(base_scores, "attrs", {}).get("test_problems", [])]
        if not test_problems:
            match = re.match(r"^prob_(.+)$", str(fold_id))
            if match:
                test_problems = [_normalize_problem_id(match.group(1))]
        if test_problems:
            return df_norm.loc[~df_norm["Problem"].isin(test_problems)].reset_index(drop=True)
        return df_norm

    if protocol == "random":
        raw_keys = getattr(base_scores, "attrs", {}).get("train_problem_dim_keys")
        if not raw_keys:
            raise RuntimeError(
                "Random-fold base scores are missing train_problem_dim_keys metadata; "
                "re-run the base-score generation step before materializing scoring variants."
            )
        key_set = {(str(problem).lower(), int(dim)) for problem, dim in raw_keys}
        mask = [
            (str(problem).lower(), int(dim)) in key_set
            for problem, dim in zip(df_norm["Problem"], df_norm["Dim"])
        ]
        return df_norm.loc[mask].reset_index(drop=True)

    return df_norm


def _materialize_predictions_from_model_scores(
    *,
    df: pd.DataFrame,
    protocols: Sequence[str],
    model_scores_by_protocol: Dict[str, Dict[str, pd.DataFrame]],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    from functions.model_interface import compute_statistical_prior, materialize_prediction_frame

    alg_cols = list(df.columns[2:])
    preds_by_protocol: Dict[str, Dict[str, pd.DataFrame]] = {}
    for protocol in protocols:
        scores_by_fold = model_scores_by_protocol.get(protocol, {})
        if not scores_by_fold:
            continue

        preds_by_fold: Dict[str, pd.DataFrame] = {}
        for fold_id, base_scores in scores_by_fold.items():
            prior = None
            if float(args.lam_prior) != 0.0:
                prior_df = _prior_source_df_for_fold(
                    df=df,
                    protocol=protocol,
                    fold_id=fold_id,
                    base_scores=base_scores,
                )
                prior = compute_statistical_prior(
                    prior_df,
                    alg_cols=alg_cols,
                    prior_scale=str(args.prior_scale),
                    sigmoid_log_s=float(args.sigmoid_log_s),
                )

            preds = materialize_prediction_frame(
                base_scores,
                alg_cols=alg_cols,
                prior=prior,
                lam_prior=float(args.lam_prior),
                tail_scale=float(args.tail_scale),
                verbose=True,
            )
            preds.attrs.update(
                {
                    "prior_scale": str(args.prior_scale),
                    "lam_prior": float(args.lam_prior),
                    "tail_scale": float(args.tail_scale),
                    "sigmoid_log_s": float(args.sigmoid_log_s),
                }
            )
            preds_by_fold[fold_id] = preds

        preds_by_protocol[protocol] = preds_by_fold

    return preds_by_protocol


def _run_orchestrator(args: argparse.Namespace) -> Dict[str, Dict[str, pd.DataFrame]]:

    from functions.model_interface import save_target_curve_plot

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_scores_dir = Path(args.base_scores_dir).expanduser().resolve() if str(args.base_scores_dir).strip() else out_dir
    base_scores_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp_tasks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = str(args.tb_log_dir).strip() if args.tb_log_dir is not None else ""
    tb_log_dir = tb_log_dir if tb_log_dir else None

    df = pd.read_csv(args.csv)
    save_target_curve_plot(
        df,
        out_dir=str(out_dir),
        target_scale=str(args.target_scale),
        sigmoid_log_s=float(args.sigmoid_log_s),
        head_2_target_scale=(str(args.head_2_target_scale or args.target_scale) if bool(args.dual_head) else None),
        head_2_sigmoid_log_s=(float(args.sigmoid_log_s) if bool(args.dual_head) else None),
    )

    protocols = _protocols_from_arg(args.protocol)

    gpus = _resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs detected; set --gpus explicitly or ensure nvidia-smi works")

    jobs_per_gpu = max(1, int(args.jobs_per_gpu))
    gpu_slots: List[Tuple[int, int]] = [(slot_id, gpu) for slot_id, gpu in enumerate(gpus * jobs_per_gpu)]

    max_parallel = int(args.max_parallel) if args.max_parallel is not None else len(gpu_slots)
    max_parallel = max(1, min(max_parallel, len(gpu_slots)))

    all_tasks: List[Task] = []
    tasks_by_protocol: Dict[str, List[Task]] = {}
    for p in protocols:
        tasks = _make_tasks(
            protocol=p,
            df=df,
            instances_all=args.instances_all,
            n_splits=args.n_splits,
        )
        all_tasks += tasks
        tasks_by_protocol[p] = tasks

    if args.dry_run:
        print(f"GPUs: {gpus} (jobs_per_gpu={jobs_per_gpu}, max_parallel={max_parallel})")
        print(f"Tasks: {len(all_tasks)}")
        by_p = {p: len(tasks_by_protocol.get(p, [])) for p in protocols}
        print("Per protocol:", json.dumps(by_p, indent=2))
        return {}

    # Simple GPU scheduler: keep <= max_parallel processes running.
    pending = all_tasks.copy()
    ctx = mp.get_context("spawn")
    running: Dict[int, Any] = {}
    running_task: Dict[int, Task] = {}
    running_gpu: Dict[int, int] = {}

    def start_task(slot_id: int, gpu: int, task: Task) -> None:
        task_out = tmp_dir / f"{task.protocol}__{task.task_id}.pkl"
        p = ctx.Process(
            target=_run_one_task,
            args=(
                gpu,
                task,
                str(task_out),
                args.csv,
                args.data_root,
                int(args.resolution),
                int(args.k_views),
                int(args.num_repetitions),
                [int(i) for i in args.instances_all],
                int(args.batch_size),
                int(args.num_epochs),
                float(args.lr),
                float(args.weight_decay),
                int(args.num_workers),
                int(args.seed),
                float(args.test_ratio),
                int(args.n_splits),
                bool(args.cache_train),
                bool(args.cache_test),
                bool(args.strict),
                bool(args.dual_head),
                float(args.sigmoid_log_s),
                str(args.target_scale),
                args.head_2_target_scale,
                float(args.head_2_loss_weight),
                float(args.head_2_score_weight),
                float(args.val_ratio_lpo),
                float(args.val_ratio_lio),
                float(args.val_ratio_random),
                int(args.early_stopping_patience),
                tb_log_dir,
                bool(args.tb_log_val),
            ),
        )
        p.start()
        running[slot_id] = p
        running_task[slot_id] = task
        running_gpu[slot_id] = gpu

    free_slots = gpu_slots[:]
    started = 0
    completed = 0

    def poll_finished() -> None:
        nonlocal completed
        finished_slots: List[int] = []
        for slot_id, proc in running.items():
            if proc.is_alive():
                continue
            task = running_task[slot_id]
            gpu = running_gpu[slot_id]
            rc = proc.exitcode
            if rc != 0:
                raise RuntimeError(
                    f"Task failed on GPU {gpu} (slot {slot_id}): {task.protocol}/{task.task_id} (exit {rc})"
                )
            finished_slots.append(slot_id)

        for slot_id in finished_slots:
            gpu = running_gpu.pop(slot_id)
            running.pop(slot_id, None)
            running_task.pop(slot_id, None)
            free_slots.append((slot_id, gpu))
            completed += 1

    print(
        f"Scheduling {len(pending)} tasks across GPUs={gpus} "
        f"(jobs_per_gpu={jobs_per_gpu}, max_parallel={max_parallel})"
    )

    while pending or running:
        poll_finished()

        while pending and free_slots and (len(running) < max_parallel):
            slot_id, gpu = free_slots.pop(0)
            task = pending.pop(0)
            start_task(slot_id, gpu, task)
            started += 1

        if running:
            time.sleep(1.0)

    print(f"All tasks completed: {completed}/{started}")

    model_scores_by_protocol: Dict[str, Dict[str, pd.DataFrame]] = {}
    for p in protocols:
        tasks = tasks_by_protocol.get(p, [])
        if not tasks:
            continue

        scores_by_fold: Dict[str, pd.DataFrame] = {}
        for task in tasks:
            task_out = tmp_dir / f"{task.protocol}__{task.task_id}.pkl"
            if task_out.exists():
                scores_by_fold[task.task_id] = pd.read_pickle(task_out)

        if not scores_by_fold:
            continue
        model_scores_by_protocol[p] = scores_by_fold

    _save_model_scores(out_dir=base_scores_dir, model_scores_by_protocol=model_scores_by_protocol)
    print(f"Saved reusable base scores to {base_scores_dir}")

    preds_by_protocol: Dict[str, Dict[str, pd.DataFrame]] = {}
    if not bool(args.base_scores_only):
        preds_by_protocol = _materialize_predictions_from_model_scores(
            df=df,
            protocols=protocols,
            model_scores_by_protocol=model_scores_by_protocol,
            args=args,
        )
        _save_fold_predictions(out_dir=out_dir, preds_by_protocol=preds_by_protocol)

        _write_protocol_outputs(
            df=df,
            out_dir=out_dir,
            protocols=protocols,
            preds_by_protocol=preds_by_protocol,
            args=args,
            tb_log_dir=tb_log_dir,
        )

    # Cleanup intermediate task outputs.
    for fp in tmp_dir.glob("*.pkl"):
        try:
            fp.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return preds_by_protocol if preds_by_protocol else model_scores_by_protocol

def _run_one_task(
    gpu: int,
    task: Task,
    out_path: str,
    csv_path: str,
    data_root: str,
    resolution: int,
    k_views: int,
    num_repetitions: int,
    instances_all: List[int],
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    test_ratio: float,
    n_splits: int,
    cache_train: bool,
    cache_test: bool,
    strict: bool,
    dual_head: bool,
    sigmoid_log_s: float,
    target_scale: str,
    head_2_target_scale: Optional[str],
    head_2_loss_weight: float,
    head_2_score_weight: float,
    val_ratio_lpo: float,
    val_ratio_lio: float,
    val_ratio_random: float,
    early_stopping_patience: int,
    tb_log_dir: Optional[str],
    tb_log_val: bool,
) -> None:
    # IMPORTANT: set CUDA_VISIBLE_DEVICES before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch

    from functions.model import GeoPAS
    from functions.model_interface import (
        MultiViewNPZDataset,
        SubsetMultiViewNPZDataset,
        _train_base_scores_from_datasets,
        _train_predict_one_split,
        default_data_dir,
        set_seed,
    )

    df = pd.read_csv(csv_path)
    alg_cols = df.columns[2:].tolist()

    def make_model():
        return GeoPAS(num_algorithms=len(alg_cols), dual_head=bool(dual_head))

    data_dir = default_data_dir(data_root, resolution)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload: Dict[str, Any] = dict(task.payload)
    instances_all = [int(i) for i in instances_all]

    set_seed(int(seed))

    tb_log_dir_use = str(tb_log_dir).strip() if tb_log_dir is not None else ""
    tb_log_dir_use = tb_log_dir_use if tb_log_dir_use else None
    tb_run_name = f"{task.protocol}/{task.task_id}"
    effective_head_2_target_scale = str(head_2_target_scale or target_scale)
    effective_head_2_loss_weight = (float(head_2_loss_weight) if bool(dual_head) else 0.0)
    effective_head_2_score_weight = (float(head_2_score_weight) if bool(dual_head) else 0.0)
    protocol_val_ratio = {
        "lpo": float(val_ratio_lpo),
        "lio": float(val_ratio_lio),
        "random": float(val_ratio_random),
    }[task.protocol]

    shared_split_kwargs: Dict[str, Any] = dict(
        data_dir=data_dir,
        num_repetitions=num_repetitions,
        k_views=k_views,
        make_model=make_model,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        cache_train=cache_train,
        cache_test=cache_test,
        strict=strict,
        target_scale=str(target_scale),
        head_2_target_scale=effective_head_2_target_scale,
        head_2_loss_weight=effective_head_2_loss_weight,
        head_2_score_weight=effective_head_2_score_weight,
        sigmoid_log_s=float(sigmoid_log_s),
        val_ratio=float(protocol_val_ratio),
        early_stopping_patience=int(early_stopping_patience),
        tb_log_dir=tb_log_dir_use,
        tb_run_name=tb_run_name,
        tb_log_val=bool(tb_log_val),
    )

    if task.protocol == "lio":
        test_inst = int(payload["test_inst"])
        train_insts = [i for i in instances_all if i != test_inst]
        out = _train_predict_one_split(
            df_train=df,
            df_test=df,
            train_instances=train_insts,
            test_instances=[test_inst],
            pbar_head=f"[train LIO i{test_inst}]",
            **shared_split_kwargs,
        )
        out.attrs["cv_protocol"] = "leave_instance_out"
        out.attrs["val_ratio"] = float(protocol_val_ratio)
        out.attrs["train_instances"] = train_insts
        out.attrs["test_instances"] = [test_inst]

    elif task.protocol == "lpo":
        test_prob = str(payload["test_prob"]).lower()
        fold_idx = int(payload.get("fold_idx", 0))
        set_seed(int(seed) + fold_idx)

        df_norm = df.copy()
        df_norm["Problem"] = df_norm["Problem"].astype(str).str.lower()
        train_df = df_norm[df_norm["Problem"] != test_prob]
        test_df = df_norm[df_norm["Problem"] == test_prob]
        if len(test_df) == 0:
            raise RuntimeError(f"No rows found for test_prob={test_prob}")

        out = _train_predict_one_split(
            df_train=train_df,
            df_test=test_df,
            train_instances=instances_all,
            test_instances=instances_all,
            pbar_head=f"[train LPO {fold_idx+1}/24]",
            **shared_split_kwargs,
        )
        out.attrs["cv_protocol"] = "leave_problem_out"
        out.attrs["val_ratio"] = float(protocol_val_ratio)
        out.attrs["test_problems"] = [test_prob]
        out.attrs["instances"] = instances_all

    elif task.protocol == "random":
        split_idx = int(payload["split_idx"])  # fold index
        k_folds = int(n_splits)
        if k_folds <= 1:
            raise ValueError(f"n_splits must be >= 2 for k-fold CV, got {k_folds}")
        if not (0 <= split_idx < k_folds):
            raise ValueError(f"split_idx out of range: {split_idx} (n_splits={k_folds})")

        # Deterministic fold partition (seeded once), fold-specific training stochasticity.
        set_seed(int(seed) + split_idx)

        base_ds = MultiViewNPZDataset(
            df,
            data_dir,
            instances=instances_all,
            num_repetitions=num_repetitions,
            cache=False,
            strict=strict,
            k_views=k_views,
            target_scale=str(target_scale),
            sigmoid_log_s=float(sigmoid_log_s),
            head_2_target_scale=effective_head_2_target_scale,
            head_2_sigmoid_log_s=float(sigmoid_log_s),
        )
        n_cases = len(base_ds)

        # Group indices by (fid, dim, instance) so all repetitions stay together.
        group_to_indices: Dict[tuple, List[int]] = {}
        for idx, m in enumerate(base_ds.meta):
            key = (int(m.fid), int(m.dim), int(m.instance))
            group_to_indices.setdefault(key, []).append(int(idx))

        group_keys = list(group_to_indices.keys())
        n_groups = len(group_keys)
        if n_groups < k_folds:
            raise RuntimeError(
                "k-fold CV requires n_groups >= n_splits; "
                f"got n_groups={n_groups}, n_splits={k_folds}."
            )

        rng_partition = np.random.default_rng(int(seed))
        perm = rng_partition.permutation(n_groups)
        fold_sizes = [n_groups // k_folds + (1 if i < (n_groups % k_folds) else 0) for i in range(k_folds)]
        folds: List[List[int]] = []
        cursor = 0
        for fs in fold_sizes:
            folds.append(perm[cursor : cursor + fs].tolist())
            cursor += fs

        test_group_ids = folds[split_idx]
        train_group_ids = [i for f in range(k_folds) if f != split_idx for i in folds[f]]

        test_idx: List[int] = []
        for gi in test_group_ids:
            test_idx.extend(group_to_indices[group_keys[gi]])
        train_idx: List[int] = []
        for gi in train_group_ids:
            train_idx.extend(group_to_indices[group_keys[gi]])

        train_ds = SubsetMultiViewNPZDataset(base_ds, train_idx, cache=cache_train)
        test_ds = SubsetMultiViewNPZDataset(base_ds, test_idx, cache=cache_test)

        fold_loader_seed = int(seed) + int(split_idx) * 1000
        train_problem_dim_keys = sorted(
            {
                (f"f{int(fid)}", int(dim))
                for fid, dim, _instance in (group_keys[gi] for gi in train_group_ids)
            }
        )

        out = _train_base_scores_from_datasets(
            train_ds=train_ds,
            test_ds=test_ds,
            alg_cols=alg_cols,
            make_model=make_model,
            device=device,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            loader_seed_base=fold_loader_seed,
            pbar_head=f"[train random s{split_idx}]",
            tb_log_dir=tb_log_dir_use,
            tb_run_name=tb_run_name,
            tb_log_val=bool(tb_log_val),
            val_ratio=float(protocol_val_ratio),
            early_stopping_patience=int(early_stopping_patience),
            head_2_loss_weight=effective_head_2_loss_weight,
            head_2_score_weight=effective_head_2_score_weight,
        )
        out.attrs["cv_protocol"] = "kfold_instance_cv"
        out.attrs["val_ratio"] = float(protocol_val_ratio)
        out.attrs["split_unit"] = "problem_dim_instance"
        out.attrs["n_folds"] = int(k_folds)
        out.attrs["fold_idx"] = int(split_idx)
        out.attrs["n_groups"] = int(n_groups)
        out.attrs["n_train_groups"] = int(len(train_group_ids))
        out.attrs["n_test_groups"] = int(len(test_group_ids))
        out.attrs["test_ratio"] = float(len(test_group_ids) / float(n_groups))
        out.attrs["n_cases"] = int(n_cases)
        out.attrs["instances"] = instances_all
        out.attrs["train_problem_dim_keys"] = train_problem_dim_keys

    else:
        raise ValueError(f"Unknown protocol: {task.protocol}")

    out.to_pickle(out_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--protocol", choices=["random", "lpo", "lio", "all"], default="all")
    p.add_argument("--csv", default="data/relert_bbob.csv")
    p.add_argument("--data-root", default="data")
    p.add_argument("--resolution", type=int, default=16)
    p.add_argument("--k-views", type=int, default=16)
    p.add_argument("--num-repetitions", type=int, default=10)
    p.add_argument("--instances-all", default="1,2,3,4,5")
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument(
        "--val-ratio-lpo",
        type=float,
        default=0.0,
        help="Training-set validation split used for early stopping during LPO folds (default: 0.0 = disabled)",
    )
    p.add_argument(
        "--val-ratio-lio",
        type=float,
        default=0.1,
        help="Training-set validation split used for early stopping during LIO folds (default: 0.1)",
    )
    p.add_argument(
        "--val-ratio-random",
        type=float,
        default=0.1,
        help="Training-set validation split used for early stopping during RANDOM folds (default: 0.1)",
    )

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    # Target transform
    p.add_argument(
        "--target-scale",
        choices=["log", "raw", "norm", "sigmoid_log", "norm_power", "log_norm_power", "log_power"],
        default="log",
        help=(
            "Training target scale: 'log' trains on log(relERT); 'raw' trains on relERT; "
            "'norm' min-max scales relERT to [0, 1]; 'sigmoid_log' applies "
            "sigma((log(relERT) - log(12.477)) / sigmoid_log_s); 'norm_power' applies "
            "norm(relERT) ^ sigmoid_log_s; 'log_norm_power' sums normalized log(relERT) and norm(relERT)."
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--aggregate-seed-dirs",
        type=str,
        default=None,
        help="Comma-separated list of seed output directories to aggregate instead of training.",
    )

    p.add_argument("--cache-train", dest="cache_train", action="store_true")
    p.add_argument("--no-cache-train", dest="cache_train", action="store_false")
    p.set_defaults(cache_train=False)

    p.add_argument("--cache-test", dest="cache_test", action="store_true")
    p.add_argument("--no-cache-test", dest="cache_test", action="store_false")
    p.set_defaults(cache_test=False)

    p.add_argument("--strict", dest="strict", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)

    p.add_argument("--gpus", default="auto", help="GPU indices to use, e.g. '0,1,2' or 'auto'")
    p.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="How many independent training jobs may share each selected GPU (default: 1)",
    )
    p.add_argument("--max-parallel", type=int, default=None)
    p.add_argument("--out-dir", default="results/bbob")
    p.add_argument(
        "--base-scores-dir",
        default="",
        help="Optional directory containing reusable base model scores, or where new base scores should be stored.",
    )
    p.add_argument("--base-scores-only", action="store_true", help="Train and save base model scores without materializing final scoring outputs.")
    p.add_argument("--materialize-only", action="store_true", help="Skip training and materialize final scoring outputs from saved base model scores.")
    p.add_argument("--dry-run", action="store_true")

    heads = p.add_mutually_exclusive_group()
    heads.add_argument("--dual-head", dest="dual_head", action="store_true", help="Use two regression heads")
    heads.add_argument("--single-head", dest="dual_head", action="store_false", help="Use a single regression head")
    p.set_defaults(dual_head=True)

    p.add_argument("--sigmoid-log-s", type=float, default=3.0)
    p.add_argument("--tail-scale", type=float, default=1.0)
    p.add_argument(
        "--lam-prior",
        type=float,
        default=0.5,
        help="Blend weight for the prior risk penalty in the final selection score.",
    )
    p.add_argument(
        "--prior-scale",
        choices=["log", "raw", "norm", "sigmoid_log", "norm_power", "log_norm_power", "log_power"],
        default="sigmoid_log",
        help="Transform used to compute the statistical prior from the fold-local training relERT table.",
    )

    p.add_argument("--head-2-loss-weight", type=float, default=0.5)
    p.add_argument("--head-2-score-weight", type=float, default=0.5)
    p.add_argument(
        "--head-2-target-scale",
        choices=["log", "raw", "norm", "sigmoid_log", "norm_power", "log_norm_power"],
        default=None,
        help="Training target scale for the optional second regression head. Defaults to --target-scale.",
    )

    p.add_argument(
        "--tb-log-dir",
        default="",
        help="TensorBoard log root directory. Empty string disables TensorBoard logging.",
    )
    p.add_argument("--tb-log-val", dest="tb_log_val", action="store_true", help="Log validation AS curve to TensorBoard")
    p.add_argument("--no-tb-log-val", dest="tb_log_val", action="store_false", help="Disable validation AS TensorBoard curve")
    p.set_defaults(tb_log_val=False)

    return p

def main():
    args = _build_parser().parse_args()
    args.instances_all = _parse_int_list(args.instances_all)

    if args.base_scores_only and args.materialize_only:
        raise ValueError("--base-scores-only and --materialize-only cannot be used together")

    for name in ("val_ratio_lpo", "val_ratio_lio", "val_ratio_random"):
        value = float(getattr(args, name))
        if not (0.0 <= value < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {value}")

    if args.aggregate_seed_dirs is not None:
        seed_dirs = [Path(s).expanduser().resolve() for s in re.split(r"[ ,]+", args.aggregate_seed_dirs) if s.strip()]
        if not seed_dirs:
            raise ValueError("--aggregate-seed-dirs was provided but no directories were parsed")

        df = pd.read_csv(args.csv)
        protocols = _protocols_from_arg(args.protocol)
        result_suffix = _result_file_suffix(args)
        seed_outputs = [
            _load_fold_predictions(out_dir=seed_dir, protocols=protocols, result_suffix=result_suffix)
            for seed_dir in seed_dirs
        ]
        aggregated_metrics = _aggregate_seed_metrics(df=df, seed_outputs=seed_outputs, protocols=protocols)
        aggregated_preds = _aggregate_seed_predictions(df=df, seed_outputs=seed_outputs, protocols=protocols)
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_fold_predictions(out_dir=out_dir, preds_by_protocol=aggregated_preds)

        tb_log_dir = str(args.tb_log_dir).strip() if args.tb_log_dir is not None else ""
        tb_log_dir = tb_log_dir if tb_log_dir else None
        _write_protocol_outputs(
            df=df,
            out_dir=out_dir,
            protocols=protocols,
            preds_by_protocol=aggregated_preds,
            metrics_by_protocol=aggregated_metrics,
            args=args,
            tb_log_dir=tb_log_dir,
        )
        print(f"Wrote aggregated outputs to {out_dir}")
        return

    if args.materialize_only:
        df = pd.read_csv(args.csv)
        protocols = _protocols_from_arg(args.protocol)
        base_scores_dir = Path(args.base_scores_dir).expanduser().resolve() if str(args.base_scores_dir).strip() else Path(args.out_dir).expanduser().resolve()
        model_scores_by_protocol = _load_model_scores(out_dir=base_scores_dir, protocols=protocols)
        if not model_scores_by_protocol:
            raise RuntimeError(f"No reusable base model scores found under {base_scores_dir}")

        preds_by_protocol = _materialize_predictions_from_model_scores(
            df=df,
            protocols=protocols,
            model_scores_by_protocol=model_scores_by_protocol,
            args=args,
        )
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_fold_predictions(out_dir=out_dir, preds_by_protocol=preds_by_protocol)

        tb_log_dir = str(args.tb_log_dir).strip() if args.tb_log_dir is not None else ""
        tb_log_dir = tb_log_dir if tb_log_dir else None
        _write_protocol_outputs(
            df=df,
            out_dir=out_dir,
            protocols=protocols,
            preds_by_protocol=preds_by_protocol,
            args=args,
            tb_log_dir=tb_log_dir,
        )
        print(f"Wrote materialized outputs to {out_dir} using base scores from {base_scores_dir}")
        return

    print(f"Resolution {args.resolution} K-Views {args.k_views} (seed={args.seed})")
    _run_orchestrator(args)


if __name__ == "__main__":
    main()
