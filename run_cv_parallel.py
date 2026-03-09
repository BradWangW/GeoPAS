#!/usr/bin/env python3
"""Parallel CV runner that schedules folds/splits across multiple GPUs.

This script is intentionally a thin orchestrator:
- The main process enumerates CV tasks (folds/splits/problems) and assigns them to GPUs.
- Each task is executed in a fresh subprocess pinned to a single GPU using CUDA_VISIBLE_DEVICES.

Supported protocols:
- random: repeated random splits over all cases (problem×dim×instance×rep)
- lpo: leave-problem-out
- lio: leave-instance-out
- inst5: 5-fold instance CV (partition instances into 5 folds)

Outputs:
- Writes one pickle per task under --out-dir.
- Writes an aggregated preds_by_fold pickle per protocol.

Run with the as_bbo env, e.g.
  /data1/home/jw1017/miniforge3/envs/as_bbo/bin/python run_cv_parallel.py --protocol all

"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _make_tasks(
    *,
    protocol: str,
    df: pd.DataFrame,
    instances_all: Sequence[int],
    n_splits: int,
    seed: int,
    shuffle_instances: bool,
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

    if protocol == "inst5":
        if len(instances_all) < 5:
            raise ValueError("inst5 requires at least 5 instances")
        insts = instances_all.copy()
        if shuffle_instances:
            rng = np.random.default_rng(seed)
            rng.shuffle(insts)
        folds = [a.tolist() for a in np.array_split(np.asarray(insts, dtype=int), 5)]
        return [
            Task(
                protocol="inst5",
                task_id=f"fold_{i}",
                payload={"fold_idx": i, "test_instances": [int(x) for x in folds[i]]},
            )
            for i in range(5)
        ]

    raise ValueError(f"Unknown protocol: {protocol}")


def _run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    protocols: List[str]
    if args.protocol == "all":
        protocols = ["random", "lpo", "lio", "inst5"]
    else:
        protocols = [args.protocol]

    gpus = _resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs detected; set --gpus explicitly or ensure nvidia-smi works")

    max_parallel = int(args.max_parallel) if args.max_parallel is not None else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))

    # Build all tasks
    all_tasks: List[Task] = []
    for p in protocols:
        all_tasks.extend(
            _make_tasks(
                protocol=p,
                df=df,
                instances_all=args.instances_all,
                n_splits=args.n_splits,
                seed=args.seed,
                shuffle_instances=args.shuffle_instances,
            )
        )

    if args.dry_run:
        print(f"GPUs: {gpus} (max_parallel={max_parallel})")
        print(f"Tasks: {len(all_tasks)}")
        by_p: Dict[str, int] = {}
        for t in all_tasks:
            by_p[t.protocol] = by_p.get(t.protocol, 0) + 1
        print("Per protocol:", json.dumps(by_p, indent=2))
        return

    # Simple GPU scheduler: keep <= max_parallel processes running, assign next task to next free gpu.
    pending = all_tasks.copy()
    running: Dict[int, subprocess.Popen] = {}
    running_task: Dict[int, Task] = {}

    def start_task(gpu: int, task: Task) -> None:
        task_out = out_dir / f"{task.protocol}__{task.task_id}.pkl"
        meta_out = out_dir / f"{task.protocol}__{task.task_id}.json"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--protocol",
            task.protocol,
            "--task-id",
            task.task_id,
            "--payload",
            json.dumps(task.payload),
            "--csv",
            args.csv,
            "--data-root",
            args.data_root,
            "--resolution",
            str(args.resolution),
            "--k-views",
            str(args.k_views),
            "--num-repetitions",
            str(args.num_repetitions),
            "--batch-size",
            str(args.batch_size),
            "--num-epochs",
            str(args.num_epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--instances-all",
            ",".join(str(i) for i in args.instances_all),
            "--test-ratio",
            str(args.test_ratio),
            "--n-splits",
            str(args.n_splits),
            "--shuffle-instances" if args.shuffle_instances else "--no-shuffle-instances",
            "--cache-train" if args.cache_train else "--no-cache-train",
            "--cache-test" if args.cache_test else "--no-cache-test",
            "--strict" if args.strict else "--no-strict",
            "--out",
            str(task_out),
            "--meta-out",
            str(meta_out),
        ]

        # Filter out the explicit no-* flags (argparse handles store_true/store_false)
        cmd = [c for c in cmd if c not in ("--no-shuffle-instances", "--no-cache-train", "--no-cache-test", "--no-strict")]

        p = subprocess.Popen(cmd, env=env)
        running[gpu] = p
        running_task[gpu] = task

    free_gpus = gpus[:]
    started = 0
    completed = 0

    def poll_finished() -> None:
        nonlocal completed
        finished_gpus: List[int] = []
        for gpu, proc in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            task = running_task[gpu]
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

    # Aggregate outputs per protocol
    for p in protocols:
        preds_by_fold: Dict[str, pd.DataFrame] = {}
        for task in [t for t in all_tasks if t.protocol == p]:
            task_out = out_dir / f"{task.protocol}__{task.task_id}.pkl"
            if not task_out.exists():
                continue
            preds_by_fold[task.task_id] = pd.read_pickle(task_out)

        if preds_by_fold:
            agg_path = out_dir / f"preds_by_fold__{p}.pkl"
            pd.to_pickle(preds_by_fold, agg_path)
            print(f"Wrote {p} aggregate: {agg_path}")


def _run_worker(args: argparse.Namespace) -> None:
    # IMPORTANT: worker is launched with CUDA_VISIBLE_DEVICES already set.
    import torch

    from functions.model import ContourCNNSelector
    from functions.model_interface import (
        MultiViewNPZDataset,
        SubsetMultiViewNPZDataset,
        _train_predict_one_split,
        default_data_dir,
        set_seed,
        single_train,
    )

    df = pd.read_csv(args.csv)
    alg_cols = df.columns[2:].tolist()

    def make_model():
        return ContourCNNSelector(num_algorithms=len(alg_cols))

    data_dir = default_data_dir(args.data_root, args.resolution)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = json.loads(args.payload)
    instances_all = [int(i) for i in _parse_int_list(args.instances_all)]

    # Make each task deterministic but independent.
    set_seed(int(args.seed))

    if args.protocol == "lio":
        test_inst = int(payload["test_inst"])
        train_insts = [i for i in instances_all if i != test_inst]
        pbar_head = f"[train LIO i{test_inst}]"
        out = _train_predict_one_split(
            df_train=df,
            df_test=df,
            data_dir=data_dir,
            train_instances=train_insts,
            test_instances=[test_inst],
            num_repetitions=args.num_repetitions,
            k_views=args.k_views,
            make_model=make_model,
            device=device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            cache_train=args.cache_train,
            cache_test=args.cache_test,
            strict=args.strict,
        )
        out.attrs["cv_protocol"] = "leave_instance_out"
        out.attrs["train_instances"] = train_insts
        out.attrs["test_instances"] = [test_inst]

    elif args.protocol == "lpo":
        test_prob = str(payload["test_prob"]).lower()
        fold_idx = int(payload.get("fold_idx", 0))
        set_seed(int(args.seed) + fold_idx)

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
            num_repetitions=args.num_repetitions,
            k_views=args.k_views,
            make_model=make_model,
            device=device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            cache_train=args.cache_train,
            cache_test=args.cache_test,
            strict=args.strict,
        )
        out.attrs["cv_protocol"] = "leave_problem_out"
        out.attrs["test_problems"] = [test_prob]
        out.attrs["instances"] = instances_all

    elif args.protocol == "random":
        split_idx = int(payload["split_idx"])
        set_seed(int(args.seed) + split_idx)

        base_ds = MultiViewNPZDataset(
            df,
            data_dir,
            instances=instances_all,
            num_repetitions=args.num_repetitions,
            cache=False,
            strict=args.strict,
            k_views=args.k_views,
        )
        n_cases = len(base_ds)
        test_size = int(np.ceil(float(args.test_ratio) * n_cases))
        test_size = max(1, min(test_size, n_cases - 1))

        rng = np.random.default_rng(int(args.seed) + split_idx)
        perm = rng.permutation(n_cases)
        test_idx = perm[:test_size].tolist()
        train_idx = perm[test_size:].tolist()

        train_ds = SubsetMultiViewNPZDataset(base_ds, train_idx, cache=args.cache_train)
        test_ds = SubsetMultiViewNPZDataset(base_ds, test_idx, cache=args.cache_test)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

        model = make_model()
        preds = single_train(
            model,
            train_loader,
            test_loader,
            device=device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pbar_head=f"[train random s{split_idx}]",
        )

        out = pd.DataFrame(preds, columns=alg_cols)
        out.insert(0, "Dim", test_ds.dims)
        out.insert(0, "Problem", test_ds.problem_ids)
        out.attrs["cv_protocol"] = "random_split_case_cv"
        out.attrs["split_idx"] = split_idx
        out.attrs["test_ratio"] = float(args.test_ratio)
        out.attrs["n_cases"] = int(n_cases)
        out.attrs["instances"] = instances_all

        del model
        torch.cuda.empty_cache()

    elif args.protocol == "inst5":
        fold_idx = int(payload["fold_idx"])
        test_instances = [int(x) for x in payload["test_instances"]]
        set_seed(int(args.seed) + fold_idx)

        test_set = set(test_instances)
        train_instances = [i for i in instances_all if i not in test_set]

        out = _train_predict_one_split(
            df_train=df,
            df_test=df,
            data_dir=data_dir,
            train_instances=train_instances,
            test_instances=test_instances,
            num_repetitions=args.num_repetitions,
            k_views=args.k_views,
            make_model=make_model,
            device=device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            cache_train=args.cache_train,
            cache_test=args.cache_test,
            strict=args.strict,
        )
        out.attrs["cv_protocol"] = "5fold_instance_cv"
        out.attrs["fold_idx"] = fold_idx
        out.attrs["train_instances"] = train_instances
        out.attrs["test_instances"] = test_instances

    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    # Persist
    pd.to_pickle(out, args.out)
    meta = {
        "protocol": args.protocol,
        "task_id": args.task_id,
        "payload": payload,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--protocol", choices=["random", "lpo", "lio", "inst5", "all"], default="all")
    p.add_argument("--csv", default="data/log_relert_bbob.csv")
    p.add_argument("--data-root", default="data")
    p.add_argument("--resolution", type=int, default=16)
    p.add_argument("--k-views", type=int, default=16)
    p.add_argument("--num-repetitions", type=int, default=10)
    p.add_argument("--instances-all", default="1,2,3,4,5")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--n-splits", type=int, default=5)

    p.add_argument("--shuffle-instances", dest="shuffle_instances", action="store_true")
    p.add_argument("--no-shuffle-instances", dest="shuffle_instances", action="store_false")
    p.set_defaults(shuffle_instances=False)

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
    p.add_argument("--out-dir", default="preds_parallel")
    p.add_argument("--dry-run", action="store_true")

    # worker mode
    p.add_argument("--worker", action="store_true")
    p.add_argument("--task-id", default="")
    p.add_argument("--payload", default="{}")
    p.add_argument("--out", default="")
    p.add_argument("--meta-out", default="")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Normalize instances in the parent; the worker reparses from string.
    if not args.worker:
        args.instances_all = _parse_int_list(args.instances_all)

    if args.worker:
        if not args.out or not args.meta_out:
            raise SystemExit("Worker mode requires --out and --meta-out")
        _run_worker(args)
    else:
        _run_orchestrator(args)


if __name__ == "__main__":
    main()
