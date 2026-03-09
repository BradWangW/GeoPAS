#!/usr/bin/env python3
"""Parallel CV runner that schedules tasks across GPUs.

Writes (per protocol) to results/bbob/<PROTOCOL>/:
- res_<res>_k_views_<k>.csv
- preds_<res>_k_views_<k>.csv.gz

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
from typing import Dict, List, Optional, Sequence

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
    payload: Dict[str, object]


def _write_metrics_csv(metrics: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("Mean\n")
        f.write("AS\n")
        metrics["scores"].to_csv(f, index=False)
        f.write("\nVBS\n")
        metrics["vbs"].to_csv(f, index=False)
        f.write("\nSBS\n")
        metrics["sbs"].to_csv(f, index=False)
        f.write("\nGap_Closure\n")
        metrics["gap_closure"].to_csv(f, index=False)

        f.write("\nMedian\n")
        f.write("AS\n")
        metrics["median_scores"].to_csv(f, index=False)
        f.write("\nVBS\n")
        metrics["median_vbs"].to_csv(f, index=False)
        f.write("\nSBS\n")
        metrics["median_sbs"].to_csv(f, index=False)
        f.write("\nGap_Closure\n")
        metrics["median_gap_closure"].to_csv(f, index=False)

        f.write("\nP90\n")
        f.write("AS\n")
        metrics["p90_scores"].to_csv(f, index=False)
        f.write("\nVBS\n")
        metrics["p90_vbs"].to_csv(f, index=False)
        f.write("\nSBS\n")
        metrics["p90_sbs"].to_csv(f, index=False)
        f.write("\nGap_Closure\n")
        metrics["p90_gap_closure"].to_csv(f, index=False)

        f.write("\nAccuracies\n")
        metrics["accuracies"].to_csv(f, index=False)

        cat_acc = metrics.get("cat_accuracies")
        if hasattr(cat_acc, "to_csv"):
            f.write("\nCatastrophe_Accuracies\n")
            cat_acc.to_csv(f, index=False)

        f.write("\nF1\n")
        metrics["f1"].to_csv(f, index=False)
        f.write("\nPick_Rate\n")
        metrics["pick_rate"].to_csv(f, header=["rate"])
        f.write("\nVBS_Pick_Rate\n")
        metrics["vbs_pick_rate"].to_csv(f, header=["rate"])


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
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        print(f"[TensorBoard] SummaryWriter unavailable; skip fold-average logging ({e}).")
        return

    metric_names = [
        "train/loss",
        "train/loss_reg",
        "train/loss_cat",
        "train/as",
        "train/lr",
        "val/loss",
        "val/loss_reg",
        "val/loss_cat",
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


def _run_orchestrator(args: argparse.Namespace) -> None:

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp_tasks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = str(args.tb_log_dir).strip() if args.tb_log_dir is not None else ""
    tb_log_dir = tb_log_dir if tb_log_dir else None

    df = pd.read_csv(args.csv)

    protocols: List[str]
    if args.protocol == "all":
        protocols = ["random", "lpo", "lio"]
    else:
        protocols = [args.protocol]

    gpus = _resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs detected; set --gpus explicitly or ensure nvidia-smi works")

    max_parallel = int(args.max_parallel) if args.max_parallel is not None else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))

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
        print(f"GPUs: {gpus} (max_parallel={max_parallel})")
        print(f"Tasks: {len(all_tasks)}")
        by_p = {p: len(tasks_by_protocol.get(p, [])) for p in protocols}
        print("Per protocol:", json.dumps(by_p, indent=2))
        return

    # Simple GPU scheduler: keep <= max_parallel processes running.
    pending = all_tasks.copy()
    ctx = mp.get_context("spawn")
    running: Dict[int, mp.Process] = {}
    running_task: Dict[int, Task] = {}

    def start_task(gpu: int, task: Task) -> None:
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
                bool(args.tail_penalty),
                float(args.tail_lam_cap),
                float(args.tail_lam_thr),
                float(args.tail_scale),
                str(args.target_scale),
                float(args.cat_loss_weight),
                float(args.cat_tau),
                float(args.cat_penalty),
                int(args.early_stopping_patience),
                tb_log_dir,
                bool(args.tb_log_val),
            ),
        )
        p.start()
        running[gpu] = p
        running_task[gpu] = task

    free_gpus = gpus[:]
    started = 0
    completed = 0

    def poll_finished() -> None:
        nonlocal completed
        finished_gpus: List[int] = []
        for gpu, proc in running.items():
            if proc.is_alive():
                continue
            task = running_task[gpu]
            rc = proc.exitcode
            if rc != 0:
                raise RuntimeError(f"Task failed on GPU {gpu}: {task.protocol}/{task.task_id} (exit {rc})")
            finished_gpus.append(gpu)

        for gpu in finished_gpus:
            running.pop(gpu, None)
            running_task.pop(gpu, None)
            free_gpus.append(gpu)
            completed += 1

    print(f"Scheduling {len(pending)} tasks across GPUs={gpus} (max_parallel={max_parallel})")

    while pending or running:
        poll_finished()

        while pending and free_gpus and (len(running) < max_parallel):
            gpu = free_gpus.pop(0)
            task = pending.pop(0)
            start_task(gpu, task)
            started += 1

        if running:
            time.sleep(1.0)

    print(f"All tasks completed: {completed}/{started}")

    from functions.model_interface import output_results

    protocol_dir = {"lpo": "LPO", "lio": "LIO", "random": "RANDOM"}
    print('Resolution', args.resolution, 'K-Views', args.k_views)
    for p in protocols:
        tasks = tasks_by_protocol.get(p, [])
        if not tasks:
            continue

        preds_by_fold: Dict[str, pd.DataFrame] = {}
        for task in tasks:
            task_out = tmp_dir / f"{task.protocol}__{task.task_id}.pkl"
            if task_out.exists():
                preds_by_fold[task.task_id] = pd.read_pickle(task_out)

        if not preds_by_fold:
            continue

        if args.tail_penalty:
            if args.dual_head:
                mid_dir = "tail_dual_head"
            else:
                mid_dir = "tail_single_head"
        else:
            if args.dual_head:
                mid_dir = "no_tail_dual_head"
            else:
                mid_dir = "no_tail_single_head"

        metrics = output_results(df, preds_by_fold, protocol=p)
        # results_dir = out_dir / protocol_dir.get(p, p) / mid_dir
        results_dir = out_dir / protocol_dir.get(p, p)
        # out_csv = results_dir / f"res_{args.resolution}_k_views_{args.k_views}.csv"

        # Name it based on TAIL_PENALTY, TAIL_LAMS_CAP, TAIL_LAMS_THR, DUAL_HEAD, CAT_LOSS_WEIGHT, CAT_TAU, CAT_PENALTY, EARLY_STOPPING_PATIENCE
        out_csv = results_dir / f"res_tailpenalty_{int(args.tail_penalty)}_taillamcap_{args.tail_lam_cap}_taillamthr_{args.tail_lam_thr}_catlossweight_{args.cat_loss_weight}_cattau_{args.cat_tau}_catpenalty_{args.cat_penalty}_res_{args.resolution}_k_views_{args.k_views}.csv"
        _write_metrics_csv(metrics, out_csv)
        print(f"Wrote {p} results CSV: {out_csv}")

        # Per-sample predictions table (concatenation of all held-out samples)
        df_preds_all = pd.concat(list(preds_by_fold.values()), ignore_index=True)
        # preds_out = results_dir / f"preds_res_{args.resolution}_k_views_{args.k_views}.csv.gz"

        preds_out = results_dir / f"preds_tailpenalty_{int(args.tail_penalty)}_taillamcap_{args.tail_lam_cap}_taillamthr_{args.tail_lam_thr}_catlossweight_{args.cat_loss_weight}_cattau_{args.cat_tau}_catpenalty_{args.cat_penalty}_res_{args.resolution}_k_views_{args.k_views}.csv.gz"

        df_preds_all.to_csv(preds_out, index=False, compression="gzip")
        print(f"Wrote {p} per-sample predictions: {preds_out}")

        _log_fold_average_curves(
            tb_log_dir=tb_log_dir,
            protocol=p,
            preds_by_fold=preds_by_fold,
            run_stub=out_dir.name,
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
    tail_penalty: bool,
    tail_lam_cap: float,
    tail_lam_thr: float,
    tail_scale: float,
    target_scale: str,
    cat_loss_weight: float,
    cat_tau: float,
    cat_penalty: float,
    early_stopping_patience: int,
    tb_log_dir: Optional[str],
    tb_log_val: bool,
) -> None:
    # IMPORTANT: set CUDA_VISIBLE_DEVICES before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch

    from functions.model import ContourCNNSelector
    from functions.model_interface import (
        MultiViewNPZDataset,
        SubsetMultiViewNPZDataset,
        _train_predict_one_split,
        default_data_dir,
        make_dataloader,
        set_seed,
        single_train,
    )

    df = pd.read_csv(csv_path)
    alg_cols = df.columns[2:].tolist()

    def make_model():
        return ContourCNNSelector(num_algorithms=len(alg_cols), dual_head=bool(dual_head))

    data_dir = default_data_dir(data_root, resolution)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = dict(task.payload)
    instances_all = [int(i) for i in instances_all]

    set_seed(int(seed))

    tb_log_dir_use = str(tb_log_dir).strip() if tb_log_dir is not None else ""
    tb_log_dir_use = tb_log_dir_use if tb_log_dir_use else None
    tb_run_name = f"{task.protocol}/{task.task_id}"

    if task.protocol == "lio":
        test_inst = int(payload["test_inst"])
        train_insts = [i for i in instances_all if i != test_inst]
        out = _train_predict_one_split(
            df_train=df,
            df_test=df,
            data_dir=data_dir,
            train_instances=train_insts,
            test_instances=[test_inst],
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
            cat_loss_weight=(float(cat_loss_weight) if bool(dual_head) else 0.0),
            cat_tau=float(cat_tau),
            cat_penalty=(float(cat_penalty) if bool(dual_head) else 0.0),
            use_tail_penalty=bool(tail_penalty),
            tail_lam_cap=float(tail_lam_cap),
            tail_lam_thr=float(tail_lam_thr),
            tail_scale=float(tail_scale),
            early_stopping_patience=int(early_stopping_patience),
            pbar_head=f"[train LIO i{test_inst}]",
            tb_log_dir=tb_log_dir_use,
            tb_run_name=tb_run_name,
            tb_log_val=bool(tb_log_val),
        )
        out.attrs["cv_protocol"] = "leave_instance_out"
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
            data_dir=data_dir,
            train_instances=instances_all,
            test_instances=instances_all,
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
            cat_loss_weight=(float(cat_loss_weight) if bool(dual_head) else 0.0),
            cat_tau=float(cat_tau),
            cat_penalty=(float(cat_penalty) if bool(dual_head) else 0.0),
            use_tail_penalty=bool(tail_penalty),
            tail_lam_cap=float(tail_lam_cap),
            tail_lam_thr=float(tail_lam_thr),
            tail_scale=float(tail_scale),
            early_stopping_patience=int(early_stopping_patience),
            pbar_head=f"[train LPO {fold_idx+1}/24]",
            tb_log_dir=tb_log_dir_use,
            tb_run_name=tb_run_name,
            tb_log_val=bool(tb_log_val),
        )
        out.attrs["cv_protocol"] = "leave_problem_out"
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
        train_loader = make_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=int(fold_loader_seed + 11),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        test_loader = make_dataloader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            seed=int(fold_loader_seed + 22),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        model = make_model()
        preds, tb_history = single_train(
            model,
            train_loader,
            test_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            pbar_head=f"[train random s{split_idx}]",
            tb_log_dir=tb_log_dir_use,
            tb_run_name=tb_run_name,
            tb_log_val=bool(tb_log_val),
            cat_loss_weight=(float(cat_loss_weight) if bool(dual_head) else 0.0),
            cat_tau=float(cat_tau),
            cat_penalty=(float(cat_penalty) if bool(dual_head) else 0.0),
            early_stopping_patience=int(early_stopping_patience),
            return_cat_arrays=False,
            return_history=True,
        )

        if bool(tail_penalty):
            from functions.model_interface import tail_table, compute_risk_penalty

            df_tail_train = tail_table(df)
            penalty = compute_risk_penalty(df_tail_train, lam_cap=float(tail_lam_cap), lam_thr=float(tail_lam_thr))
            preds = preds + float(tail_scale) * penalty[None, :]

        out = pd.DataFrame(preds, columns=alg_cols)
        out.insert(0, "Repetition", test_ds.repetitions)
        out.insert(0, "Instance", test_ds.instance_ids)
        out.insert(0, "Dim", test_ds.dims)
        out.insert(0, "Problem", test_ds.problem_ids)
        out.attrs["tb_history"] = tb_history
        out.attrs["cv_protocol"] = "kfold_instance_cv"
        out.attrs["split_unit"] = "problem_dim_instance"
        out.attrs["n_folds"] = int(k_folds)
        out.attrs["fold_idx"] = int(split_idx)
        out.attrs["n_groups"] = int(n_groups)
        out.attrs["n_train_groups"] = int(len(train_group_ids))
        out.attrs["n_test_groups"] = int(len(test_group_ids))
        out.attrs["test_ratio"] = float(len(test_group_ids) / float(n_groups))
        out.attrs["n_cases"] = int(n_cases)
        out.attrs["instances"] = instances_all

        del model
        torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown protocol: {task.protocol}")

    pd.to_pickle(out, out_path)


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

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    # Target transform
    p.add_argument(
        "--target-scale",
        choices=["log", "raw"],
        default="log",
        help="Training target scale: 'log' trains on log(relERT); 'raw' trains on relERT.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--n-splits", type=int, default=5)

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
    p.add_argument("--max-parallel", type=int, default=None)
    p.add_argument("--out-dir", default="results/bbob")
    p.add_argument("--dry-run", action="store_true")

    heads = p.add_mutually_exclusive_group()
    heads.add_argument("--dual-head", dest="dual_head", action="store_true", help="Use regression + catastrophe heads")
    heads.add_argument("--single-head", dest="dual_head", action="store_false", help="Use regression-only head")
    p.set_defaults(dual_head=True)

    tp = p.add_mutually_exclusive_group()
    tp.add_argument("--tail-penalty", dest="tail_penalty", action="store_true", help="Add per-algorithm tail risk penalty")
    tp.add_argument("--no-tail-penalty", dest="tail_penalty", action="store_false", help="Disable tail risk penalty")
    p.set_defaults(tail_penalty=False)

    p.add_argument("--tail-lam-cap", type=float, default=15.0)
    p.add_argument("--tail-lam-thr", type=float, default=3.0)
    p.add_argument("--tail-scale", type=float, default=1.0)

    p.add_argument("--cat-loss-weight", type=float, default=15.0)
    p.add_argument("--cat-tau", type=float, default=0.5)
    p.add_argument("--cat-penalty", type=float, default=15.0)

    p.add_argument(
        "--tb-log-dir",
        default="",
        help="TensorBoard log root directory. Empty string disables TensorBoard logging.",
    )
    p.add_argument("--tb-log-val", dest="tb_log_val", action="store_true", help="Log validation AS curve to TensorBoard")
    p.add_argument("--no-tb-log-val", dest="tb_log_val", action="store_false", help="Disable validation AS TensorBoard curve")
    p.set_defaults(tb_log_val=False)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.instances_all = _parse_int_list(args.instances_all)

    print('Resolution', args.resolution, 'K-Views', args.k_views)

    _run_orchestrator(args)


if __name__ == "__main__":
    main()
