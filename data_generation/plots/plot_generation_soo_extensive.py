from itertools import product
import numpy as np
import os
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from auxiliary_functions import process_problem_chunk
import json

warnings.filterwarnings(
    "ignore",
    message="The balance properties of Sobol' points require n to be a power of 2",
    category=UserWarning,
)

# ---------------- Configuration ---------------- #

num_repetitions = 10
dims = [2, 3, 5, 10]
func_ids = range(1, 25)
instances = range(1, 6)

resolutions = [8, 16, 32, 64]
k_views = 128

ell_min = 0.02
ell_max = 0.7  
log_uniform_scale = True

use_processes = True
n_workers = max(1, mp.cpu_count() - 1)

out_dir = f"data/bbob/maxscale_{ell_max}_logscale_{str(log_uniform_scale).lower()}"

print(f"Repetitions: {num_repetitions}")
print(f"Workers: {n_workers} processes")

# BBOB grouping
group_map = {
    1: range(1, 6),
    2: range(6, 10),
    3: range(10, 15),
    4: range(15, 20),
    5: range(20, 25),
}
func_to_group = {fid: g for g, fs in group_map.items() for fid in fs}

# ---------------- Execution ---------------- #

for resolution in tqdm(
    resolutions,
    desc="(resolution, k)",
    total=len(resolutions),
):

    out_dir_res = f"{out_dir}/res_{resolution}"
    # out_dir_res = f"data/bbob_uniform/res_{resolution}"
    os.makedirs(out_dir_res, exist_ok=True)

    tasks = []
    for dim in dims:
        for fid in func_ids:
            for inst in instances:
                tasks.append((fid, inst, dim))

    print(f"\nRunning res={resolution}, k={k_views} with {len(tasks)} tasks")

    timing_records = {}
    timing_by_dim = {}
    timing_by_func_dim = {}
    start_total = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                process_problem_chunk,
                fid,
                inst,
                dim,
                num_repetitions,
                k_views,
                resolution,
                ell_min,
                ell_max,
                log_uniform_scale,
                out_dir_res,
                func_to_group,
            )
            for (fid, inst, dim) in tasks
        ]

        for future, (fid, inst, dim) in zip(
            tqdm(futures, desc="Collecting", leave=False),
            tasks,
        ):
            try:
                timing_data = future.result()
                for ((timed_fid, timed_dim), t) in timing_data:
                    group_id = func_to_group.get(timed_fid)
                    if group_id is not None:
                        timing_records.setdefault((group_id, timed_dim), []).append(t)
                    timing_by_dim.setdefault(timed_dim, []).append(t)
                    timing_by_func_dim.setdefault((timed_fid, timed_dim), []).append(t)
            except Exception as e:
                print(f"Error in f{fid}_i{inst}_dim{dim}: {e}")

    elapsed = time.time() - start_total
    print(f"Total elapsed: {elapsed:.1f}s")

    def _stats(ts):
        arr = np.asarray(ts, dtype=float)
        if arr.size == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    avg_time_per_dim = {str(dim): _stats(ts) for dim, ts in sorted(timing_by_dim.items())}
    avg_time_per_func_dim = {
        f"f{fid}_dim_{dim}": _stats(ts)
        for (fid, dim), ts in sorted(timing_by_func_dim.items())
    }

    timing_summary = {
        f"group_{group}_dim_{dim}": {**_stats(ts), "times": [float(x) for x in ts]}
        for (group, dim), ts in sorted(timing_records.items())
    }

    timing_out = {
        "resolution": resolution,
        "k_views": k_views,
        "num_repetitions": num_repetitions,
        "dims": list(dims),
        "func_ids": [int(x) for x in func_ids],
        "instances": [int(x) for x in instances],
        "n_tasks": len(tasks),
        "n_workers": n_workers,
        "elapsed_total_sec": float(elapsed),
        "avg_time_sec_per_dim": avg_time_per_dim,
        "avg_time_sec_per_func_dim": avg_time_per_func_dim,
        "timings_group_dim": timing_summary,
    }

    timing_path = os.path.join(out_dir, f"timings_res_{resolution}.json")
    with open(timing_path, "w") as f:
        json.dump(timing_out, f, indent=2)

    for group in range(1, 6):
        for dim in (2, 10):
            ts = timing_records.get((group, dim), [])
            if ts:
                print(
                    f"Group {group}, dim={dim}: "
                    f"avg {np.mean(ts):.3f}s (n={len(ts)})"
                )
