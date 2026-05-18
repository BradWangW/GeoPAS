import os
import re
import copy
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ----------------------------- utilities ---------------------------------- #

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers) and (int(num_workers) > 0),
        worker_init_fn=_seed_worker,
        generator=g,
    )


def problem_to_fid(problem: str | int) -> int:
    """Accepts 1, '1', 'f1', 'F1' -> 1."""
    if isinstance(problem, (int, np.integer)):
        return int(problem)
    s = str(problem).strip()
    m = re.match(r"^[fF]?(\d+)$", s)
    if not m:
        raise ValueError(f"Unrecognised Problem id: {problem}")
    return int(m.group(1))


def default_data_dir(root: str, resolution: int) -> str:
    # matches the generator layout: data/bbob/res_{resolution}
    return os.path.join(root, f"res_{resolution}")


TARGET_SCALE_CHOICES = {"log", "raw", "norm", "sigmoid_log", "norm_power", "log_norm_power", "log_power"}
TARGET_SCALES_NEED_BOUNDS = {"norm", "norm_power", "log_norm_power"}


def _resolve_target_scale(target_scale: Optional[str], *, fallback: str = "log") -> str:
    scale = fallback if target_scale is None else str(target_scale).lower().strip()
    if scale not in TARGET_SCALE_CHOICES:
        raise ValueError(
            f"target_scale must be one of {sorted(TARGET_SCALE_CHOICES)}, got: {target_scale!r}"
        )
    return scale


def _transform_target_np(
    target: np.ndarray,
    *,
    target_scale: str,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
    sigmoid_log_s: float = 1.2,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    if target_scale == "log":
        log_target = np.log(np.maximum(target, 1e-6)).astype(np.float32)
        norm_log_target = log_target / np.max(log_target) 
        return norm_log_target
    if target_scale == "log_power":
        log_target = np.log(np.maximum(target, 1e-6)).astype(np.float32)
        norm_log_target = log_target / np.max(log_target) 
        return np.power(norm_log_target, sigmoid_log_s)
    if target_scale == "sigmoid_log":
        log_target = np.log(np.maximum(target, 1e-6))
        s = float(sigmoid_log_s)
        return (1.0 / (1.0 + np.exp( -(log_target - np.log(14.863))/s ))).astype(np.float32)
    if target_scale in TARGET_SCALES_NEED_BOUNDS:
        if target_min is None or target_max is None:
            raise ValueError(
                "target_min and target_max are required when target_scale is 'norm', 'norm_power', or 'log_norm_power'"
            )
        scale = max(float(target_max) - float(target_min), 1e-6)
        norm_target = ((target - float(target_min)) / scale).astype(np.float32)
        if target_scale == "norm_power":
            return np.power(norm_target, float(sigmoid_log_s)).astype(np.float32)
        if target_scale == "log_norm_power":
            log_target = np.log(np.maximum(target, 1e-6)).astype(np.float32)
            return ((1 - sigmoid_log_s) * log_target / np.max(log_target) + sigmoid_log_s * norm_target).astype(np.float32)
        return norm_target
    return target.astype(np.float32)


def _transform_target_rows_np(
    target: np.ndarray,
    *,
    target_scale: str,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
    sigmoid_log_s: float = 1.2,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim <= 1:
        return _transform_target_np(
            target,
            target_scale=target_scale,
            target_min=target_min,
            target_max=target_max,
            sigmoid_log_s=sigmoid_log_s,
        )

    rows = [
        _transform_target_np(
            row,
            target_scale=target_scale,
            target_min=target_min,
            target_max=target_max,
            sigmoid_log_s=sigmoid_log_s,
        )
        for row in target
    ]
    if not rows:
        return np.empty_like(target, dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def save_target_curve_plot(
    df: pd.DataFrame,
    *,
    out_dir: str,
    target_scale: str,
    sigmoid_log_s: float = 1.2,
    head_2_target_scale: Optional[str] = None,
    head_2_sigmoid_log_s: Optional[float] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[target-plot] Skipping target plot because matplotlib is unavailable: {exc}")
        return

    alg_cols = list(df.columns[2:])
    raw_values = df.loc[:, alg_cols].to_numpy(dtype=np.float32, copy=False)
    raw_min = max(float(np.min(raw_values)), 1e-6)
    raw_max = max(float(np.max(raw_values)), raw_min * 1.01)
    x_values = np.geomspace(raw_min, raw_max, num=512).astype(np.float32)

    target_scale = _resolve_target_scale(target_scale)
    target_min = raw_min if target_scale in TARGET_SCALES_NEED_BOUNDS else None
    target_max = raw_max if target_scale in TARGET_SCALES_NEED_BOUNDS else None
    y_main = _transform_target_np(
        x_values,
        target_scale=target_scale,
        target_min=target_min,
        target_max=target_max,
        sigmoid_log_s=float(sigmoid_log_s),
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(x_values, y_main, linewidth=2.0, label=f"main: {target_scale}")

    ax.plot(x_values, x_values/np.max(x_values), linewidth=1.0, linestyle=":", label="identity")
    ax.plot(x_values, np.log(np.maximum(x_values, 1e-6))/np.log(np.max(np.maximum(x_values, 1e-6))), linewidth=1.0, linestyle=":", label="log")

    if head_2_target_scale is not None:
        head_2_target_scale = _resolve_target_scale(head_2_target_scale, fallback=target_scale)
        head_2_sigmoid_log_s_f = float(head_2_sigmoid_log_s) if head_2_sigmoid_log_s is not None else float(sigmoid_log_s)
        head_2_target_min = raw_min if head_2_target_scale in TARGET_SCALES_NEED_BOUNDS else None
        head_2_target_max = raw_max if head_2_target_scale in TARGET_SCALES_NEED_BOUNDS else None

        if not (
            head_2_target_scale == target_scale
            and head_2_target_min == target_min
            and head_2_target_max == target_max
            and head_2_sigmoid_log_s_f == float(sigmoid_log_s)
        ):
            y_head_2 = _transform_target_np(
                x_values,
                target_scale=head_2_target_scale,
                target_min=head_2_target_min,
                target_max=head_2_target_max,
                sigmoid_log_s=head_2_sigmoid_log_s_f,
            )
            ax.plot(x_values, y_head_2, linewidth=2.0, linestyle="--", label=f"head_2: {head_2_target_scale}")

    ax.set_xscale("log")
    ax.set_xlabel("raw relERT")
    ax.set_ylabel("transformed target")
    ax.set_title("Target Transform Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_curve.png"), dpi=200)
    plt.close(fig)


def compute_statistical_prior(
    df_train: pd.DataFrame,
    *,
    alg_cols: Optional[Sequence[str]] = None,
    prior_scale: str = "sigmoid_log",
    sigmoid_log_s: float = 1.2,
) -> np.ndarray:
    alg_cols = list(df_train.columns[2:] if alg_cols is None else alg_cols)
    target = df_train.loc[:, alg_cols].to_numpy(dtype=np.float32, copy=False)
    if target.size == 0:
        return np.zeros(len(alg_cols), dtype=np.float64)

    prior_scale = _resolve_target_scale(prior_scale)
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    if prior_scale in TARGET_SCALES_NEED_BOUNDS:
        target_min = float(np.min(target))
        target_max = float(np.max(target))

    transformed = _transform_target_rows_np(
        target,
        target_scale=prior_scale,
        target_min=target_min,
        target_max=target_max,
        sigmoid_log_s=float(sigmoid_log_s),
    )
    prior = np.asarray(np.mean(transformed, axis=0, dtype=np.float64), dtype=np.float64)
    print(
        f"Statistical prior ({prior_scale}):\n"
        f"{pd.Series(prior, index=alg_cols).round(3)}",
        flush=True,
    )
    return prior


# ----------------------------- dataset ------------------------------------ #

@dataclass(frozen=True)
class SampleMeta:
    fid: int
    dim: int
    instance: int
    rep: int


class MultiViewNPZDataset(Dataset):
    """
    Each item loads a single .npz containing:
      - plots: (K, r, r) float32
            - masks: (K, r, r) uint8 or bool (0/1)   (legacy key: 'mask')
            - stats: (K, 3)    float32               (legacy key: 'ell' -> stats[:,0])
    Returns:
      plots  (K,r,r) float32
      masks  (K,r,r) float32 in {0,1}
            stats  (K,3)   float32
            target (M,)    float32  (main-head algorithm performance vector)
            head_2_target (M,) float32  (second-head algorithm performance vector)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        *,
        instances: Sequence[int],
        num_repetitions: int,
        cache: bool = False,
        strict: bool = True,
        k_views: int = 32,
        target_scale: str = "log",
        target_min: Optional[float] = None,
        target_max: Optional[float] = None,
        sigmoid_log_s: float = 1.2,
        head_2_target_scale: Optional[str] = None,
        head_2_target_min: Optional[float] = None,
        head_2_target_max: Optional[float] = None,
        head_2_sigmoid_log_s: Optional[float] = None,
    ):
        # Keep a private copy to avoid mutating caller data.
        # (Important: evaluation expects relERT >= 1; training may use log-targets.)
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = data_dir
        self.instances = list(instances)
        self.num_repetitions = int(num_repetitions)
        self.cache = bool(cache)
        self.strict = bool(strict)
        self.k_views = int(k_views)
        self.target_scale = _resolve_target_scale(target_scale)
        self.head_2_target_scale = _resolve_target_scale(head_2_target_scale, fallback=self.target_scale)

        # targets are all columns after Problem, Dim
        self.alg_cols = list(self.df.columns[2:])
        self.target_min = float(target_min) if target_min is not None else None
        self.target_max = float(target_max) if target_max is not None else None
        self.sigmoid_log_s = float(sigmoid_log_s)
        if self.target_scale in TARGET_SCALES_NEED_BOUNDS and (self.target_min is None or self.target_max is None):
            target_values = self.df.loc[:, self.alg_cols].to_numpy(dtype=np.float32, copy=False)
            self.target_min = float(np.min(target_values))
            self.target_max = float(np.max(target_values))

        self.head_2_target_min = float(head_2_target_min) if head_2_target_min is not None else None
        self.head_2_target_max = float(head_2_target_max) if head_2_target_max is not None else None
        self.head_2_sigmoid_log_s = (
            float(head_2_sigmoid_log_s) if head_2_sigmoid_log_s is not None else self.sigmoid_log_s
        )
        if self.head_2_target_scale in TARGET_SCALES_NEED_BOUNDS and (
            self.head_2_target_min is None or self.head_2_target_max is None
        ):
            target_values = self.df.loc[:, self.alg_cols].to_numpy(dtype=np.float32, copy=False)
            self.head_2_target_min = float(np.min(target_values))
            self.head_2_target_max = float(np.max(target_values))

        # Build index: one file per (row in df) × instance × repetition
        self.files: List[str] = []
        self.meta: List[SampleMeta] = []
        self.targets: List[torch.Tensor] = []
        self.head_2_targets: List[torch.Tensor] = []

        # Optional in-memory cache
        self._cache_plots: List[torch.Tensor] = []
        self._cache_masks: List[torch.Tensor] = []
        self._cache_stats: List[torch.Tensor] = []
        self._cache_dims: List[int] = []

        for row in range(len(self.df)):
            fid = problem_to_fid(self.df.loc[row, "Problem"])
            dim = int(self.df.loc[row, "Dim"])

            target_raw = self.df.loc[row, self.alg_cols].astype(np.float32).to_numpy()
            target_np = _transform_target_np(
                target_raw,
                target_scale=self.target_scale,
                target_min=self.target_min,
                target_max=self.target_max,
                sigmoid_log_s=self.sigmoid_log_s,
            )
            target = torch.from_numpy(target_np)
            if (
                self.head_2_target_scale == self.target_scale
                and self.head_2_target_min == self.target_min
                and self.head_2_target_max == self.target_max
                and self.head_2_sigmoid_log_s == self.sigmoid_log_s
            ):
                head_2_target = target
            else:
                head_2_target_np = _transform_target_np(
                    target_raw,
                    target_scale=self.head_2_target_scale,
                    target_min=self.head_2_target_min,
                    target_max=self.head_2_target_max,
                    sigmoid_log_s=self.head_2_sigmoid_log_s,
                )
                head_2_target = torch.from_numpy(head_2_target_np)

            for inst in self.instances:
                for rep in range(self.num_repetitions):
                    tag = f"f{fid}_i{inst}_dim{dim}_rep{rep}.npz"
                    path = os.path.join(self.data_dir, tag)

                    if not os.path.exists(path):
                        if self.strict:
                            raise FileNotFoundError(path)
                        else:
                            continue

                    self.files.append(path)
                    self.meta.append(SampleMeta(fid=fid, dim=dim, instance=inst, rep=rep))
                    self.targets.append(target)
                    self.head_2_targets.append(head_2_target)

                    if self.cache:
                        arr = np.load(path)
                        
                        # Take the first k_views only
                        plots_np = arr["plots"][:self.k_views]
                        masks_np = arr["masks"][:self.k_views]
                        stats_np = arr["stats"][:self.k_views]

                        plots = torch.from_numpy(plots_np).to(torch.float32)
                        mask = torch.from_numpy(masks_np).to(torch.float32)  # {0,1}
                        stats = torch.from_numpy(stats_np).to(torch.float32)
                        self._cache_plots.append(plots)
                        self._cache_masks.append(mask)
                        self._cache_stats.append(stats)
                        self._cache_dims.append(dim)

        if len(self.files) == 0:
            raise RuntimeError(f"No samples found in {data_dir} for instances={instances}.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if self.cache:
            plots = self._cache_plots[idx]
            masks = self._cache_masks[idx]
            stats = self._cache_stats[idx]
            dim = self._cache_dims[idx]
        else:
            arr = np.load(self.files[idx])
            plots = torch.from_numpy(arr["plots"][:self.k_views]).to(torch.float32)
            masks = torch.from_numpy(arr["masks"][:self.k_views]).to(torch.float32)  # {0,1}
            stats = torch.from_numpy(arr["stats"][:self.k_views]).to(torch.float32)
            dim = self.meta[idx].dim

        target = self.targets[idx]
        head_2_target = self.head_2_targets[idx]
        return plots, masks, stats, dim, target, head_2_target

    @property
    def problem_ids(self) -> List[str]:
        return [f"f{m.fid}" for m in self.meta]

    @property
    def dims(self) -> List[int]:
        return [m.dim for m in self.meta]

    @property
    def instance_ids(self) -> List[int]:
        return [m.instance for m in self.meta]

    @property
    def repetitions(self) -> List[int]:
        return [m.rep for m in self.meta]


class SubsetMultiViewNPZDataset(Dataset):
    """Subset view of a `MultiViewNPZDataset` with optional per-subset caching."""

    def __init__(
        self,
        base: MultiViewNPZDataset,
        indices: Sequence[int],
        *,
        cache: bool = False,
    ):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.cache = bool(cache)

        if len(self.indices) == 0:
            raise ValueError("Subset indices must be non-empty")

        if self.cache:
            self._cache_items = [self.base[base_idx] for base_idx in self.indices]
        else:
            self._cache_items = []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.cache:
            return self._cache_items[idx]
        return self.base[self.indices[idx]]

    def _meta_values(self, field: str) -> List[int]:
        return [int(getattr(self.base.meta[i], field)) for i in self.indices]

    @property
    def problem_ids(self) -> List[str]:
        return [f"f{fid}" for fid in self._meta_values("fid")]

    @property
    def dims(self) -> List[int]:
        return self._meta_values("dim")

    @property
    def instance_ids(self) -> List[int]:
        return self._meta_values("instance")

    @property
    def repetitions(self) -> List[int]:
        return self._meta_values("rep")


# -------------------------- training / evaluation -------------------------- #

def _sanitize_tb_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9._\-/]+", "", s)
    s = s.strip("._/")
    return s or "run"


def _make_tb_run_dir(tb_log_dir: str, tb_run_name: Optional[str], pbar_head: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_tb_name(tb_run_name) if tb_run_name else _sanitize_tb_name(pbar_head)
    return os.path.join(tb_log_dir, f"{base}__{stamp}")


def _pairwise_logistic_ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss within each sample.

    Assumptions:
    - pred, target are shape (B, M)
    - lower values mean better

    For each unordered pair (a,b), we compute s = sign(target_b - target_a).
    If s==+1 then b is worse than a, so we want pred_b - pred_a > 0.
    Loss per pair: softplus(-s * (pred_b - pred_a)).
    Ties (s==0) are ignored.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"pred/target must be 2D (B,M), got {pred.shape} and {target.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target must have same shape, got {pred.shape} vs {target.shape}")

    bsz, n_alg = pred.shape
    if n_alg < 2 or bsz == 0:
        return pred.new_zeros(())

    idx = torch.triu_indices(n_alg, n_alg, offset=1, device=pred.device)
    a, b = idx[0], idx[1]

    pred_diff = pred[:, b] - pred[:, a]
    with torch.no_grad():
        s = torch.sign(target[:, b] - target[:, a])
        m = s != 0

    if not torch.any(m):
        return pred.new_zeros(())
    return F.softplus(-s[m] * pred_diff[m]).mean()

    
def cvar_loss(per_sample_loss, alpha=0.9):
    """
    per_sample_loss: (B,) tensor
    alpha: tail fraction (e.g., 0.9 keeps worst 10%)
    """
    B = per_sample_loss.shape[0]
    k = int((1 - alpha) * B)
    if k < 1:
        return per_sample_loss.mean()

    worst, _ = torch.topk(per_sample_loss, k=k, largest=True)
    return worst.mean()


def asymmetric_mse(pred, target, alpha=5.0):
    """
    alpha > 1 penalises underestimation more
    """
    diff = pred - target
    loss = diff ** 2
    weight = torch.where(diff < 0, alpha, 1.0)
    return (weight * loss).mean()


def percentile_weighted_loss(pred, target):
    diff = torch.abs(pred - target)
    ranks = torch.argsort(torch.argsort(target))
    weights = 1.0 + ranks.float() / ranks.max()
    return (weights * diff).mean()


def _blend_dual_head_outputs(
    main_output: torch.Tensor,
    head_2_output: Optional[torch.Tensor],
    *,
    head_2_weight: float,
) -> torch.Tensor:
    if head_2_output is None:
        return main_output
    weight = float(head_2_weight)
    return (1.0 - weight) * main_output + weight * head_2_output


def single_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pbar_head: str = "[train]",
    tb_log_dir: Optional[str] = None,
    tb_run_name: Optional[str] = None,
    tb_log_val: bool = False,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    return_history: bool = False,
    val_ratio: float = 0.0,
    early_stopping_patience: int = 15,
) -> Any:
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_decay_gamma = 1.0 / (1.0 + 100.0 * max(0.0, float(weight_decay)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_gamma)

    reg_loss_fn = nn.SmoothL1Loss()
    head_2_loss_weight_f = float(head_2_loss_weight)
    head_2_score_weight_f = float(head_2_score_weight)

    base_loader_seed = int(torch.initial_seed()) & 0xFFFFFFFF

    def _make_loader_like(base: DataLoader, dataset: Dataset, *, shuffle: bool, seed_offset: int) -> DataLoader:
        return make_dataloader(
            dataset,
            batch_size=int(base.batch_size),
            shuffle=shuffle,
            num_workers=int(base.num_workers),
            seed=int(base_loader_seed + seed_offset),
            pin_memory=bool(getattr(base, "pin_memory", False)),
            persistent_workers=bool(getattr(base, "persistent_workers", False)),
        )

    # Split the *training* dataset into train/val for early stopping.
    val_loader: Optional[DataLoader] = None
    train_es_loader = train_loader
    try:
        n_total = len(train_loader.dataset)
    except Exception:
        n_total = 0

    if float(val_ratio) > 0.0 and n_total >= 2:
        n_val = int(round(float(val_ratio) * n_total))
        n_val = max(1, min(n_val, n_total - 1))
        n_train = n_total - n_val

        # Use the current torch RNG seed (already set by set_seed in callers).
        g = torch.Generator()
        g.manual_seed(int(torch.initial_seed()) & 0xFFFFFFFF)
        train_ds, val_ds = torch.utils.data.random_split(train_loader.dataset, [n_train, n_val], generator=g)

        train_es_loader = _make_loader_like(train_loader, train_ds, shuffle=True, seed_offset=101)
        val_loader = _make_loader_like(train_loader, val_ds, shuffle=False, seed_offset=202)

    def _forward(plots: torch.Tensor, masks: torch.Tensor, stats: torch.Tensor, dim: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = model(plots, masks, stats, dim)
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError(f"Model must return (pred, head_2_pred|None), got {type(out)} of len {len(out)}")
            pred, head_2_pred = out
            return pred, head_2_pred
        return out, None

    writer = None
    if tb_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TensorBoard logging requested but unavailable. "
                "Install it via conda/pip (e.g. `conda install -c conda-forge tensorboard`)."
            ) from e
        run_dir = _make_tb_run_dir(tb_log_dir, tb_run_name, pbar_head)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)

    try:
        pbar = tqdm(range(num_epochs), desc=f"{pbar_head} epoch", unit="epoch")

        history: Dict[str, List[float]] = {
            "train/loss": [],
            "train/loss_main": [],
            "train/loss_head_2": [],
            "train/as": [],
            "train/lr": [],
            "val/loss": [],
            "val/loss_main": [],
            "val/loss_head_2": [],
            "val/as": [],
        }

        best_val: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        for epoch in pbar:
            epoch_losses = []
            epoch_main_losses = []
            epoch_head_2_losses = []
            epoch_train_as_total = 0.0
            epoch_train_as_count = 0
            for step, (plots, masks, stats, dim, target, head_2_target) in enumerate(train_es_loader):
                plots = plots.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                dim = dim.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                head_2_target = head_2_target.to(device, non_blocking=True)
                
                optim.zero_grad(set_to_none=True)
                pred, head_2_pred = _forward(plots, masks, stats, dim)

                loss_main = reg_loss_fn(pred, target)
                if head_2_pred is None:
                    loss_head_2 = pred.new_zeros(())
                    loss = loss_main
                else:
                    loss_head_2 = reg_loss_fn(head_2_pred, head_2_target)
                    loss = (1.0 - head_2_loss_weight_f) * loss_main + head_2_loss_weight_f * loss_head_2

                train_scores = _blend_dual_head_outputs(
                    pred,
                    head_2_pred,
                    head_2_weight=head_2_score_weight_f,
                )
                train_targets = _blend_dual_head_outputs(
                    target,
                    head_2_target if head_2_pred is not None else None,
                    head_2_weight=head_2_score_weight_f,
                )
                train_pick = torch.argmin(train_scores, dim=1)
                train_achieved = train_targets.gather(1, train_pick.unsqueeze(1)).squeeze(1)
                epoch_train_as_total += float(train_achieved.sum().item())
                epoch_train_as_count += int(train_achieved.numel())
                
                loss.backward()
                optim.step()

                epoch_losses.append(loss.item())
                epoch_main_losses.append(float(loss_main.detach().item()))
                epoch_head_2_losses.append(float(loss_head_2.detach().item()))

            # Validation score for early stopping (mean objective, lower is better).
            val_score: Optional[float] = None
            val_loss: Optional[float] = None
            val_loss_main: Optional[float] = None
            val_loss_head_2: Optional[float] = None
            if val_loader is not None:
                model.eval()
                total = 0
                total_as = 0.0
                total_val_loss = 0.0
                total_val_loss_main = 0.0
                total_val_loss_head_2 = 0.0
                with torch.no_grad():
                    for plots, masks, stats, dim, target, head_2_target in val_loader:
                        plots = plots.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        stats = stats.to(device, non_blocking=True)
                        dim = dim.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        head_2_target = head_2_target.to(device, non_blocking=True)
                        pred, head_2_pred = _forward(plots, masks, stats, dim)

                        v_loss_main = reg_loss_fn(pred, target)
                        if head_2_pred is None:
                            v_loss_head_2 = pred.new_zeros(())
                            v_loss = v_loss_main
                        else:
                            v_loss_head_2 = reg_loss_fn(head_2_pred, head_2_target)
                            v_loss = (1.0 - head_2_loss_weight_f) * v_loss_main + head_2_loss_weight_f * v_loss_head_2

                        scores = _blend_dual_head_outputs(
                            pred,
                            head_2_pred,
                            head_2_weight=head_2_score_weight_f,
                        )
                        score_targets = _blend_dual_head_outputs(
                            target,
                            head_2_target if head_2_pred is not None else None,
                            head_2_weight=head_2_score_weight_f,
                        )

                        pick = torch.argmin(scores, dim=1)  # (B,)
                        achieved = score_targets.gather(1, pick.unsqueeze(1)).squeeze(1)  # (B,)
                        bs = int(achieved.numel())
                        total_as += float(achieved.sum().item())
                        total_val_loss += float(v_loss.item()) * bs
                        total_val_loss_main += float(v_loss_main.item()) * bs
                        total_val_loss_head_2 += float(v_loss_head_2.item()) * bs
                        total += bs

                if total > 0:
                    val_score = float(total_as / total)
                    val_loss = float(total_val_loss / total)
                    val_loss_main = float(total_val_loss_main / total)
                    val_loss_head_2 = float(total_val_loss_head_2 / total)
                    if writer is not None and tb_log_val:
                        writer.add_scalar("val/as", val_score, epoch)
                        writer.add_scalar("val/loss", val_loss, epoch)
                        writer.add_scalar("val/loss_main", val_loss_main, epoch)
                        writer.add_scalar("val/loss_head_2", val_loss_head_2, epoch)
                model.train()

            # update the epoch progress bar
            if epoch_losses:
                avg_loss = float(sum(epoch_losses) / len(epoch_losses))
                avg_main = float(sum(epoch_main_losses) / len(epoch_main_losses)) if epoch_main_losses else 0.0
                avg_head_2 = float(sum(epoch_head_2_losses) / len(epoch_head_2_losses)) if epoch_head_2_losses else 0.0
                train_as = float(epoch_train_as_total / epoch_train_as_count) if epoch_train_as_count > 0 else float("nan")
                if val_score is None:
                    pbar.set_postfix(main=f"{avg_main:.6f}", h2=f"{avg_head_2:.6f}", train_as=f"{train_as:.6f}")
                else:
                    pbar.set_postfix(main=f"{avg_main:.6f}", h2=f"{avg_head_2:.6f}", train_as=f"{train_as:.6f}", val=f"{val_score:.6f}")
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, epoch)
                    writer.add_scalar("train/loss_main", avg_main, epoch)
                    writer.add_scalar("train/loss_head_2", avg_head_2, epoch)
                    writer.add_scalar("train/as", train_as, epoch)
                    writer.add_scalar("train/lr", float(optim.param_groups[0]["lr"]), epoch)

                history["train/loss"].append(float(avg_loss))
                history["train/loss_main"].append(float(avg_main))
                history["train/loss_head_2"].append(float(avg_head_2))
                history["train/as"].append(float(train_as))
                history["train/lr"].append(float(optim.param_groups[0]["lr"]))
                history["val/loss"].append(float(val_loss) if val_loss is not None else float("nan"))
                history["val/loss_main"].append(float(val_loss_main) if val_loss_main is not None else float("nan"))
                history["val/loss_head_2"].append(float(val_loss_head_2) if val_loss_head_2 is not None else float("nan"))
                history["val/as"].append(float(val_score) if val_score is not None else float("nan"))

                scheduler.step()

            # Early stopping based on validation score.
            if val_score is not None and int(early_stopping_patience) > 0 and epoch >= 10:
                if best_val is None or val_score < best_val:
                    best_val = val_score
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(early_stopping_patience):
                        break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    # evaluate
    model.eval()
    preds_all = []
    with torch.no_grad():
        for plots, masks, stats, dim, _, _ in test_loader:
            plots = plots.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            dim   = dim.to(device, non_blocking=True)

            pred, head_2_pred = _forward(plots, masks, stats, dim)
            scores = _blend_dual_head_outputs(
                pred,
                head_2_pred,
                head_2_weight=head_2_score_weight_f,
            )
            preds_all.append(scores.detach().cpu().numpy())

    preds = np.concatenate(preds_all, axis=0)

    if return_history:
        return preds, history

    return preds


def _prediction_frame_from_scores(
    *,
    scores: np.ndarray,
    test_ds: Any,
    alg_cols: Sequence[str],
    attrs: Optional[Mapping[Hashable | None, Any]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(np.asarray(scores), columns=list(alg_cols))
    out.insert(0, "Repetition", test_ds.repetitions)
    out.insert(0, "Instance", test_ds.instance_ids)
    out.insert(0, "Dim", test_ds.dims)
    out.insert(0, "Problem", test_ds.problem_ids)
    out.attrs.update(dict(attrs or {}))
    return out


def _print_score_examples(
    *,
    base_scores: np.ndarray,
    final_scores: np.ndarray,
    prior: Optional[np.ndarray],
    lam_prior: float,
) -> None:
    if prior is None:
        print("Pred examples:")
        for i in range(min(5, final_scores.shape[0])):
            pred_row = np.array2string(final_scores[i], precision=3, suppress_small=True)
            print(f"  Pred: {pred_row}", flush=True)
        return

    print("Base score and prior examples:")
    prior_row = np.array2string(prior, precision=3, suppress_small=True)
    for i in range(min(5, final_scores.shape[0])):
        base_row = np.array2string(base_scores[i], precision=3, suppress_small=True)
        score_row = np.array2string(final_scores[i], precision=3, suppress_small=True)
        print(
            f"  Base: {base_row} | Prior: {prior_row} | lam_prior={lam_prior:g} => Score: {score_row}",
            flush=True,
        )


def materialize_prediction_frame(
    base_scores: pd.DataFrame,
    *,
    alg_cols: Optional[Sequence[str]] = None,
    prior: Optional[np.ndarray] = None,
    lam_prior: float = 0.0,
    tail_scale: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    alg_cols = list(base_scores.columns[4:] if alg_cols is None else alg_cols)
    meta_cols = [col for col in ("Problem", "Dim", "Instance", "Repetition") if col in base_scores.columns]
    meta = base_scores.loc[:, meta_cols].copy()
    base_values = base_scores.loc[:, alg_cols].to_numpy(dtype=np.float64, copy=True)
    final_scores = base_values
    scaled_prior: Optional[np.ndarray] = None

    if prior is not None:
        scaled_prior = float(tail_scale) * np.asarray(prior, dtype=np.float64)
        final_scores = (1.0 - float(lam_prior)) * base_values + float(lam_prior) * scaled_prior[None, :]

    if verbose:
        _print_score_examples(
            base_scores=base_values,
            final_scores=final_scores,
            prior=scaled_prior,
            lam_prior=float(lam_prior),
        )

    out = pd.concat(
        [meta, pd.DataFrame(final_scores, columns=alg_cols, index=meta.index)],
        axis=1,
    )
    out.attrs.update(dict(getattr(base_scores, "attrs", {})))
    return out


def _train_base_scores_from_datasets(
    *,
    train_ds: Dataset,
    test_ds: Dataset,
    alg_cols: Sequence[str],
    make_model: Callable[[], nn.Module],
    device: torch.device,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    loader_seed_base: int,
    pbar_head: str,
    tb_log_dir: Optional[str],
    tb_run_name: Optional[str],
    tb_log_val: bool,
    val_ratio: float,
    early_stopping_patience: int,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
) -> pd.DataFrame:
    train_loader = make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=int(loader_seed_base + 11),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        seed=int(loader_seed_base + 22),
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
        pbar_head=pbar_head,
        tb_log_dir=tb_log_dir,
        tb_run_name=tb_run_name,
        tb_log_val=tb_log_val,
        val_ratio=float(val_ratio),
        head_2_loss_weight=float(head_2_loss_weight),
        head_2_score_weight=float(head_2_score_weight),
        return_history=True,
        early_stopping_patience=int(early_stopping_patience),
    )

    out = _prediction_frame_from_scores(
        scores=preds,
        test_ds=test_ds,
        alg_cols=alg_cols,
        attrs={"tb_history": tb_history},
    )

    del model
    torch.cuda.empty_cache()
    return out


def _train_predict_from_datasets(
    *,
    train_ds: Dataset,
    test_ds: Dataset,
    alg_cols: Sequence[str],
    make_model: Callable[[], nn.Module],
    device: torch.device,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    loader_seed_base: int,
    pbar_head: str,
    tb_log_dir: Optional[str],
    tb_run_name: Optional[str],
    tb_log_val: bool,
    val_ratio: float,
    early_stopping_patience: int,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    penalty: Optional[np.ndarray] = None,
    lam_prior: float = 0.5,
) -> pd.DataFrame:
    base_scores = _train_base_scores_from_datasets(
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
        loader_seed_base=loader_seed_base,
        pbar_head=pbar_head,
        tb_log_dir=tb_log_dir,
        tb_run_name=tb_run_name,
        tb_log_val=tb_log_val,
        val_ratio=val_ratio,
        early_stopping_patience=early_stopping_patience,
        head_2_loss_weight=head_2_loss_weight,
        head_2_score_weight=head_2_score_weight,
    )
    return materialize_prediction_frame(
        base_scores,
        alg_cols=alg_cols,
        prior=penalty,
        lam_prior=float(lam_prior),
        tail_scale=1.0,
        verbose=True,
    )


def _train_predict_one_split(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    data_dir: str,
    train_instances: Sequence[int],
    test_instances: Sequence[int],
    num_repetitions: int,
    k_views: int,
    make_model: Callable[[], nn.Module],
    device: torch.device,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    cache_train: bool,
    cache_test: bool,
    strict: bool,
    target_scale: str = "log",
    head_2_target_scale: Optional[str] = None,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    sigmoid_log_s: float = 3.0,
    pbar_head: str = "[train]",
    tb_log_dir: Optional[str] = None,
    tb_run_name: Optional[str] = None,
    tb_log_val: bool = False,
    val_ratio: float = 0.0,
    early_stopping_patience: int = 15,
) -> pd.DataFrame:
    alg_cols = list(df_train.columns[2:])
    target_scale = _resolve_target_scale(target_scale)
    head_2_target_scale = _resolve_target_scale(head_2_target_scale, fallback=target_scale)

    def _resolve_target_bounds(scale: str) -> Tuple[Optional[float], Optional[float]]:
        if scale not in TARGET_SCALES_NEED_BOUNDS:
            return None, None
        target_values = pd.concat(
            [df_train.loc[:, alg_cols], df_test.loc[:, alg_cols]],
            axis=0,
            ignore_index=True,
        ).to_numpy(dtype=np.float32, copy=False)
        return float(np.min(target_values)), float(np.max(target_values))

    target_min, target_max = _resolve_target_bounds(target_scale)
    head_2_target_min, head_2_target_max = _resolve_target_bounds(head_2_target_scale)
    sigmoid_log_s = float(sigmoid_log_s)

    train_ds = MultiViewNPZDataset(
        df_train,
        data_dir,
        instances=train_instances,
        num_repetitions=num_repetitions,
        cache=cache_train,
        strict=strict,
        k_views=k_views,
        target_scale=target_scale,
        target_min=target_min,
        target_max=target_max,
        sigmoid_log_s=sigmoid_log_s,
        head_2_target_scale=head_2_target_scale,
        head_2_target_min=head_2_target_min,
        head_2_target_max=head_2_target_max,
        head_2_sigmoid_log_s=sigmoid_log_s,
    )
    test_ds = MultiViewNPZDataset(
        df_test,
        data_dir,
        instances=test_instances,
        num_repetitions=num_repetitions,
        cache=cache_test,
        strict=strict,
        k_views=k_views,
        target_scale=target_scale,
        target_min=target_min,
        target_max=target_max,
        sigmoid_log_s=sigmoid_log_s,
        head_2_target_scale=head_2_target_scale,
        head_2_target_min=head_2_target_min,
        head_2_target_max=head_2_target_max,
        head_2_sigmoid_log_s=sigmoid_log_s,
    )

    return _train_base_scores_from_datasets(
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
        loader_seed_base=int(torch.initial_seed()) & 0xFFFFFFFF,
        pbar_head=pbar_head,
        tb_log_dir=tb_log_dir,
        tb_run_name=tb_run_name,
        tb_log_val=tb_log_val,
        val_ratio=float(val_ratio),
        early_stopping_patience=int(early_stopping_patience),
        head_2_loss_weight=float(head_2_loss_weight),
        head_2_score_weight=float(head_2_score_weight),
    )

def run_random_split(
    df: pd.DataFrame,
    *,
    data_root: str,
    resolution: int,
    k_views: int,
    num_repetitions: int,
    instances_all: Sequence[int] = (1, 2, 3, 4, 5),
    make_model,  # callable: () -> nn.Module
    n_splits: int = 5,
    test_ratio: float = 0.2,
    batch_size: int = 16,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    cache_train: bool = False,
    cache_test: bool = False,
    strict: bool = True,
    target_scale: str = "log",
    head_2_target_scale: Optional[str] = None,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    target_transform_scale: float = 1.2,
    seed: int = 42,
    tb_log_dir: Optional[str] = None,
    tb_log_val: bool = False,
    val_ratio: float = 0.1,
    early_stopping_patience: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Random CV over (problem×dim×instance) groups (repetitions kept together).

    - We build a base dataset of *cases* (problem×dim×instance×rep).
    - We then group indices by (problem, dim, instance) so all repetitions for a
      given instance stay in the same fold.
    - If n_splits >= 2, we run a proper K-fold CV where the K test folds are a
      partition of the full dataset (at the group level).

    Note: `test_ratio` is only used when n_splits == 1 (single holdout).
    """
    if int(n_splits) <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")
    if int(n_splits) == 1 and not (0.0 < float(test_ratio) < 1.0):
        raise ValueError(f"test_ratio must be in (0,1) when n_splits==1, got {test_ratio}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = default_data_dir(data_root, resolution)

    instances_all = [int(i) for i in instances_all]
    if len(instances_all) == 0:
        raise ValueError("instances_all must be non-empty")
    if len(set(instances_all)) != len(instances_all):
        raise ValueError(f"instances_all contains duplicates: {instances_all}")

    # Build a single shared index of all cases, then subset it per fold.
    base_ds = MultiViewNPZDataset(
        df,
        data_dir,
        instances=instances_all,
        num_repetitions=num_repetitions,
        cache=False,
        strict=strict,
        k_views=k_views,
        target_scale=target_scale,
        sigmoid_log_s=float(target_transform_scale),
        head_2_target_scale=head_2_target_scale,
        head_2_sigmoid_log_s=float(target_transform_scale),
    )
    n_cases = len(base_ds)
    if n_cases < 2:
        raise RuntimeError("Random split requires at least 2 total cases")

    # Group indices by (fid, dim, instance) so all repetitions stay together.
    group_to_indices: Dict[tuple, List[int]] = {}
    for idx, m in enumerate(base_ds.meta):
        key = (int(m.fid), int(m.dim), int(m.instance))
        group_to_indices.setdefault(key, []).append(int(idx))

    group_keys = list(group_to_indices.keys())
    n_groups = len(group_keys)
    if n_groups < 2:
        raise RuntimeError("Random split requires at least 2 total (problem,dim,instance) groups")

    k_folds = int(n_splits)
    if k_folds >= 2 and n_groups < k_folds:
        raise RuntimeError(
            "k-fold CV requires n_groups >= n_splits; "
            f"got n_groups={n_groups}, n_splits={k_folds}."
        )

    # One deterministic shuffle of groups; fold assignment is fixed by `seed`.
    rng_partition = np.random.default_rng(int(seed))
    perm = rng_partition.permutation(n_groups)
    if k_folds >= 2:
        fold_sizes = [n_groups // k_folds + (1 if i < (n_groups % k_folds) else 0) for i in range(k_folds)]
        folds: List[List[int]] = []
        cursor = 0
        for fs in fold_sizes:
            folds.append(perm[cursor : cursor + fs].tolist())
            cursor += fs
    else:
        # Single holdout split at the GROUP level.
        test_groups_n = int(np.ceil(float(test_ratio) * n_groups))
        test_groups_n = max(1, min(test_groups_n, n_groups - 1))
        folds = [perm[:test_groups_n].tolist()]

    preds_by_fold: Dict[str, pd.DataFrame] = {}
    for fold_idx in range(int(n_splits)):
        
        if k_folds >= 2:
            test_group_ids = folds[fold_idx]
            train_group_ids = [i for f in range(k_folds) if f != fold_idx for i in folds[f]]
        else:
            test_group_ids = folds[0]
            train_group_ids = [i for i in perm.tolist() if i not in set(test_group_ids)]

        test_idx: List[int] = []
        for gi in test_group_ids:
            test_idx.extend(group_to_indices[group_keys[gi]])
        train_idx: List[int] = []
        for gi in train_group_ids:
            train_idx.extend(group_to_indices[group_keys[gi]])

        train_ds = SubsetMultiViewNPZDataset(base_ds, train_idx, cache=cache_train)
        test_ds = SubsetMultiViewNPZDataset(base_ds, test_idx, cache=cache_test)

        out = _train_predict_from_datasets(
            train_ds=train_ds,
            test_ds=test_ds,
            alg_cols=list(df.columns[2:]),
            make_model=make_model,
            device=device,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            loader_seed_base=int(seed) + int(fold_idx) * 1000,
            pbar_head=f"[train kfold(inst) {fold_idx+1}/{n_splits}]" if k_folds >= 2 else "[train holdout(inst)]",
            tb_log_dir=tb_log_dir,
            tb_run_name=f"random/fold_{fold_idx}",
            tb_log_val=tb_log_val,
            val_ratio=float(val_ratio),
            early_stopping_patience=int(early_stopping_patience),
            head_2_loss_weight=float(head_2_loss_weight),
            head_2_score_weight=float(head_2_score_weight),
            penalty=None,
        )

        out.attrs["cv_protocol"] = "kfold_instance_cv" if k_folds >= 2 else "holdout_instance_cv"
        out.attrs["split_unit"] = "problem_dim_instance"
        out.attrs["n_folds"] = int(k_folds)
        out.attrs["fold_idx"] = int(fold_idx)
        out.attrs["n_groups"] = int(n_groups)
        out.attrs["n_train_groups"] = int(len(train_group_ids))
        out.attrs["n_test_groups"] = int(len(test_group_ids))
        out.attrs["test_ratio"] = float(len(test_group_ids) / float(n_groups))
        out.attrs["n_cases"] = int(n_cases)
        out.attrs["n_train_cases"] = int(len(train_ds))
        out.attrs["n_test_cases"] = int(len(test_ds))
        out.attrs["instances"] = instances_all
        preds_by_fold[f"split_{fold_idx}"] = out

    return preds_by_fold

def run_leave_problem_out(
    df: pd.DataFrame,
    *,
    data_root: str,
    resolution: int,
    k_views: int,
    num_repetitions: int,
    instances_all: Sequence[int] = (1, 2, 3, 4, 5),
    problems_all: Optional[Sequence[int | str]] = None,
    make_model,  # callable: () -> nn.Module
    batch_size: int = 16,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    cache_train: bool = False,
    cache_test: bool = False,
    strict: bool = True,
    target_scale: str = "log",
    head_2_target_scale: Optional[str] = None,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    target_transform_scale: float = 1.2,
    seed: int = 42,
    tb_log_dir: Optional[str] = None,
    tb_log_val: bool = False,
    val_ratio: float = 0.0,
    early_stopping_patience: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Leave-problem-out (LPO): one held-out BBOB function id per fold."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = default_data_dir(data_root, resolution)

    instances_all = [int(i) for i in instances_all]
    if len(instances_all) == 0:
        raise ValueError("instances_all must be non-empty")
    if len(set(instances_all)) != len(instances_all):
        raise ValueError(f"instances_all contains duplicates: {instances_all}")

    if problems_all is None:
        problems_all_norm = sorted(pd.unique(df["Problem"].astype(str).str.lower()))
        # Keep only patterns like f<number> if present.
        problems_all = problems_all_norm

    # Normalize to canonical string keys used by the dataset ('f1', 'f2', ...)
    problems_all_norm2: List[str] = []
    for p in problems_all:
        fid = problem_to_fid(p)
        problems_all_norm2.append(f"f{fid}")

    preds_by_fold: Dict[str, pd.DataFrame] = {}
    df_norm = df.copy()
    df_norm["Problem"] = df_norm["Problem"].astype(str).str.lower()

    present = set(pd.unique(df_norm["Problem"]))
    test_probs = [p for p in problems_all_norm2 if p in present]
    if len(test_probs) == 0:
        raise RuntimeError("No LPO folds were generated; check problems_all and df['Problem']")

    for fold_idx, test_prob in enumerate(test_probs):
        set_seed(seed + fold_idx)
        train_df = df_norm[df_norm["Problem"] != test_prob]
        test_df = df_norm[df_norm["Problem"] == test_prob]

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
            target_scale=target_scale,
            head_2_target_scale=head_2_target_scale,
            head_2_loss_weight=float(head_2_loss_weight),
            head_2_score_weight=float(head_2_score_weight),
            sigmoid_log_s=float(target_transform_scale),
            pbar_head=f"[train LPO {fold_idx+1}/{len(test_probs)}]",
            tb_log_dir=tb_log_dir,
            tb_run_name=f"lpo/{test_prob}",
            tb_log_val=tb_log_val,
            val_ratio=float(val_ratio),
            early_stopping_patience=int(early_stopping_patience)
        )
        out.attrs["cv_protocol"] = "leave_problem_out"
        out.attrs["train_problems"] = sorted(pd.unique(train_df["Problem"]))
        out.attrs["test_problems"] = [test_prob]
        out.attrs["instances"] = instances_all
        preds_by_fold[f"prob_{test_prob}"] = out

    if len(preds_by_fold) == 0:
        raise RuntimeError("No LPO folds were generated; check problems_all and df['Problem']")

    return preds_by_fold

def run_leave_instance_out(
    df: pd.DataFrame,
    *,
    data_root: str,
    resolution: int,
    k_views: int,
    num_repetitions: int,
    instances_all: Sequence[int] = (1, 2, 3, 4, 5),
    make_model,  # callable: () -> nn.Module, captures num_algorithms etc.
    batch_size: int = 16,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    cache_train: bool = False,
    cache_test: bool = False,
    strict: bool = True,
    target_scale: str = "log",
    head_2_target_scale: Optional[str] = None,
    head_2_loss_weight: float = 0.5,
    head_2_score_weight: float = 0.5,
    target_transform_scale: float = 1.2,
    seed: int = 42,
    tb_log_dir: Optional[str] = None,
    tb_log_val: bool = False,
    val_ratio: float = 0.1,
    early_stopping_patience: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Leave-instance-out (LIO) protocol: one held-out instance per fold."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = default_data_dir(data_root, resolution)

    preds_by_fold: Dict[str, pd.DataFrame] = {}

    instances_all = [int(i) for i in instances_all]
    if len(instances_all) == 0:
        raise ValueError("instances_all must be non-empty")
    if len(set(instances_all)) != len(instances_all):
        raise ValueError(f"instances_all contains duplicates: {instances_all}")

    for fold, test_inst in enumerate(instances_all):
        train_insts = [i for i in instances_all if i != test_inst]

        train_ds = MultiViewNPZDataset(
            df,
            data_dir,
            instances=train_insts,
            num_repetitions=num_repetitions,
            cache=cache_train,
            strict=strict,
            k_views=k_views,
            target_scale=target_scale,
            sigmoid_log_s=float(target_transform_scale),
            head_2_target_scale=head_2_target_scale,
            head_2_sigmoid_log_s=float(target_transform_scale),
        )
        test_ds = MultiViewNPZDataset(
            df,
            data_dir,
            instances=[test_inst],
            num_repetitions=num_repetitions,
            cache=cache_test,
            strict=strict,
            k_views=k_views,
            target_scale=target_scale,
            sigmoid_log_s=float(target_transform_scale),
            head_2_target_scale=head_2_target_scale,
            head_2_sigmoid_log_s=float(target_transform_scale),
        )

        out = _train_predict_from_datasets(
            train_ds=train_ds,
            test_ds=test_ds,
            alg_cols=list(df.columns[2:]),
            make_model=make_model,
            device=device,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            loader_seed_base=int(seed) + int(fold) * 1000,
            pbar_head=f"[train LIO {fold+1}/{len(instances_all)}]",
            tb_log_dir=tb_log_dir,
            tb_run_name=f"lio/inst_{test_inst}",
            tb_log_val=tb_log_val,
            val_ratio=float(val_ratio),
            early_stopping_patience=int(early_stopping_patience),
            head_2_loss_weight=float(head_2_loss_weight),
            head_2_score_weight=float(head_2_score_weight),
            penalty=None,
        )
        out.attrs["cv_protocol"] = "leave_instance_out"
        out.attrs["train_instances"] = train_insts
        out.attrs["test_instances"] = [test_inst]
        preds_by_fold[f"fold_{fold}"] = out

    return preds_by_fold


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Sequence[str]) -> float:
    """Macro-F1 over a fixed label set, with 0 for undefined labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    f1s: List[float] = []
    for lab in labels:
        tp = float(np.sum((y_true == lab) & (y_pred == lab)))
        fp = float(np.sum((y_true != lab) & (y_pred == lab)))
        fn = float(np.sum((y_true == lab) & (y_pred != lab)))
        denom = (2.0 * tp + fp + fn)
        f1s.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def output_results(
    df: pd.DataFrame,
    dict_df_predictions: Dict[str, pd.DataFrame],
    protocol: Optional[str] = None,
    print_fold_summary: bool = True,
) -> Dict[str, Any]:
    """
    Compute evaluation summaries from per-fold prediction DataFrames.

    Inputs
    - df: ground-truth relERT table with columns [Problem, Dim, <alg1>, <alg2>, ...]
    - dict_df_predictions: {fold_id: predictions_df} where predictions_df has columns
      [Problem, Dim, <alg1>, <alg2>, ...] (lower predicted value => better)

    Returns
    A dict with keys:
    - scores: mean true relERT achieved by the picked algorithm
    - median_scores: median true relERT achieved by the picked algorithm (computed on concatenation of all folds)
    - p90_scores: 90th percentile true relERT achieved by the picked algorithm (computed on concatenation of all folds)
    - accuracies: exact-match accuracy vs the true best algorithm
    - f1: macro-F1 over algorithms
    - pick_rate: algorithm pick frequency
    - vbs_pick_rate: oracle pick frequency (true best algorithm distribution)
    - sbs: SBS (single best solver) true relERT baseline
    - vbs: VBS (virtual best solver) true relERT baseline
    - gap_closure: VBS–SBS gap closure of the model selection
    - median_sbs/median_vbs/median_gap_closure
    - p90_sbs/p90_vbs/p90_gap_closure
    """
    alg_cols = list(df.columns[2:])

    # Normalize ground truth keys to match prediction formatting.
    df_gt = df.copy()
    df_gt["Problem"] = df_gt["Problem"].astype(str).str.lower()
    df_gt["Dim"] = df_gt["Dim"].astype(int)

    # Sanity check: relERT should be >= 1. If we see values < 1, the caller likely
    # passed a log-transformed table (e.g., log_relert_bbob.csv), which will produce
    # AS values < 1.
    try:
        gt_vals = df_gt[alg_cols].to_numpy(dtype=float)
        min_val = float(np.nanmin(gt_vals)) if gt_vals.size else float("nan")
        if np.isfinite(min_val) and min_val < 1.0:
            frac_lt1 = float(np.mean(gt_vals < 1.0))
            print(
                "WARNING: output_results(): ground truth has values < 1.0 "
                f"(min={min_val:.6g}, frac<1={frac_lt1:.3%}). "
                "If this is meant to be relERT, you likely passed a log-domain CSV.",
                flush=True,
            )
    except Exception:
        pass

    # Ground-truth performance matrix keyed by (Problem, Dim)
    perf_by_key = df_gt.set_index(["Problem", "Dim"])[alg_cols]
    true_best_alg = perf_by_key.idxmin(axis=1).rename("true_alg")

    # SBS definition (global): choose ONE algorithm over the entire dataset `df`.
    # Then report its performance on each subgroup/dimension subset.
    perf_all = perf_by_key.to_numpy(dtype=float)
    sbs_alg_idx_global = int(np.argmin(np.mean(perf_all, axis=0)))
    sbs_alg_idx_med_global = sbs_alg_idx_global
    sbs_alg_idx_p90_global = sbs_alg_idx_global

    # BBOB groups
    group_defs = [
        ("f1-f5", 1, 5),
        ("f6-f9", 6, 9),
        ("f10-f14", 10, 14),
        ("f15-f19", 15, 19),
        ("f20-f24", 20, 24),
    ]
    group_names = [g for (g, _, _) in group_defs] + ["all"]

    # Dimensions shown in the summary tables.
    # Use ground-truth `df` (protocol-invariant) rather than the union of
    # held-out folds (which can differ across CV protocols / repeated splits).
    dims = sorted(pd.unique(df_gt["Dim"].astype(int)))
    all_dims: List[object] = list(dims) + ["all"]

    col_to_idx = {c: i for i, c in enumerate(alg_cols)}

    dict_df_scores: Dict[str, pd.DataFrame] = {}
    dict_df_accuracies: Dict[str, pd.DataFrame] = {}
    dict_df_f1: Dict[str, pd.DataFrame] = {}
    dict_df_sbs: Dict[str, pd.DataFrame] = {}
    dict_df_vbs: Dict[str, pd.DataFrame] = {}
    dict_df_gap: Dict[str, pd.DataFrame] = {}

    for fold_id, preds in dict_df_predictions.items():
        preds = preds.copy()
        preds.attrs = {}
        preds["Problem"] = preds["Problem"].astype(str).str.lower()
        preds["Dim"] = preds["Dim"].astype(int)

        pred_alg = preds[alg_cols].idxmin(axis=1).to_numpy()
        keys = pd.MultiIndex.from_frame(preds[["Problem", "Dim"]])

        # True relERT row for each sample, then pick the column of the predicted algorithm.
        perf_rows = perf_by_key.loc[keys].to_numpy(dtype=float)
        pred_idx = pd.Series(pred_alg).map(col_to_idx).to_numpy(dtype=int)
        picked_score = perf_rows[np.arange(len(perf_rows)), pred_idx]

        sbs_alg_idx = sbs_alg_idx_global

        true_alg = true_best_alg.loc[keys].to_numpy()
        correct = (pred_alg == true_alg)

        # Assign group label per row
        fid = preds["Problem"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
        group = np.full(fid.shape, "all", dtype=object)
        for name, lo, hi in group_defs:
            group[(fid >= lo) & (fid <= hi)] = name

        # Aggregate
        # IMPORTANT: use NaN for (dim,group) pairs with no samples in this fold.
        # Otherwise, averaging across folds (e.g., LPO) is diluted by zeros.
        performances = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        accuracies = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        f1_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        sbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        vbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        gap_closure = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        dims_arr = preds["Dim"].to_numpy()

        for di, dval in enumerate(all_dims):
            dim_mask = np.ones(len(preds), dtype=bool) if dval == "all" else (dims_arr == int(dval))
            for gi, gname in enumerate(group_names):
                grp_mask = np.ones(len(preds), dtype=bool) if gname == "all" else (group == gname)
                m = dim_mask & grp_mask
                if not np.any(m):
                    continue
                as_score = float(np.mean(picked_score[m]))
                performances[di, gi] = as_score
                accuracies[di, gi] = float(np.mean(correct[m]))
                f1_scores[di, gi] = _macro_f1(true_alg[m], pred_alg[m], labels=alg_cols)

                # Baselines on the same subset
                subset_perf = perf_rows[m]
                vbs = float(np.mean(np.min(subset_perf, axis=1)))
                sbs = float(np.mean(subset_perf[:, sbs_alg_idx]))
                vbs_scores[di, gi] = vbs
                sbs_scores[di, gi] = sbs

                denom = (sbs - vbs)
                if abs(denom) <= 1e-12:
                    gap_closure[di, gi] = 1.0 if abs(as_score - vbs) <= 1e-12 else 0.0
                else:
                    gap_closure[di, gi] = (sbs - as_score) / denom

        df_scores = pd.DataFrame(performances, columns=group_names)
        df_scores.insert(0, "Dim", all_dims)
        df_acc = pd.DataFrame(accuracies, columns=group_names)
        df_acc.insert(0, "Dim", all_dims)
        df_f1 = pd.DataFrame(f1_scores, columns=group_names)
        df_f1.insert(0, "Dim", all_dims)
        df_sbs = pd.DataFrame(sbs_scores, columns=group_names)
        df_sbs.insert(0, "Dim", all_dims)
        df_vbs = pd.DataFrame(vbs_scores, columns=group_names)
        df_vbs.insert(0, "Dim", all_dims)
        df_gap = pd.DataFrame(gap_closure, columns=group_names)
        df_gap.insert(0, "Dim", all_dims)

        if print_fold_summary:
            as_mean = float(np.mean(picked_score))
            as_median = float(np.median(picked_score))
            as_p90 = float(np.quantile(picked_score, 0.9))
            print(
                f"Fold-local overall for {fold_id}, "
                f"AS mean: {as_mean:.3f}, AS median: {as_median:.3f}, AS P90: {as_p90:.3f}"
            )

        dict_df_scores[fold_id] = df_scores.round(3)
        dict_df_accuracies[fold_id] = df_acc.round(3)
        dict_df_f1[fold_id] = df_f1.round(3)
        dict_df_sbs[fold_id] = df_sbs.round(3)
        dict_df_vbs[fold_id] = df_vbs.round(3)
        dict_df_gap[fold_id] = df_gap.round(3)

    def _compute_sbs_vbs_on_full_df(
        *,
        use_median: bool = False,
        quantile: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute SBS/VBS baselines on the full ground-truth df (protocol-invariant).

        SBS is defined by selecting ONE global algorithm on the full df and then
        reporting its aggregated relERT on each subgroup subset.

        VBS is computed per subset by taking the per-row best (min over
        algorithms) then aggregating.
        """
        if quantile is not None:
            q = float(quantile)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"quantile must be in [0,1], got {quantile}")
            if use_median:
                raise ValueError("use_median and quantile are mutually exclusive")
        gt = perf_by_key.reset_index().copy()  # columns: Problem, Dim, <alg...>
        gt["Problem"] = gt["Problem"].astype(str).str.lower()
        gt["Dim"] = gt["Dim"].astype(int)

        perf = gt[alg_cols].to_numpy(dtype=float)
        perf_metric = perf
        if quantile is not None:
            sbs_alg_idx = int(sbs_alg_idx_p90_global)
        else:
            sbs_alg_idx = int(sbs_alg_idx_med_global) if use_median else int(sbs_alg_idx_global)

        fid = gt["Problem"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
        group = np.full(fid.shape, "all", dtype=object)
        for name, lo, hi in group_defs:
            group[(fid >= lo) & (fid <= hi)] = name

        dims_arr = gt["Dim"].to_numpy(dtype=int)
        sbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        vbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)

        for di, dval in enumerate(all_dims):
            dim_mask = np.ones(len(gt), dtype=bool) if dval == "all" else (dims_arr == int(dval))
            for gi, gname in enumerate(group_names):
                grp_mask = np.ones(len(gt), dtype=bool) if gname == "all" else (group == gname)
                m = dim_mask & grp_mask
                if not np.any(m):
                    continue

                subset_perf = perf_metric[m]
                row_min = np.min(subset_perf, axis=1)
                if quantile is not None:
                    vbs_scores[di, gi] = float(np.quantile(row_min, q))
                    sbs_scores[di, gi] = float(np.quantile(subset_perf[:, sbs_alg_idx], q))
                elif use_median:
                    vbs_scores[di, gi] = float(np.median(row_min))
                    sbs_scores[di, gi] = float(np.median(subset_perf[:, sbs_alg_idx]))
                else:
                    vbs_scores[di, gi] = float(np.mean(row_min))
                    sbs_scores[di, gi] = float(np.mean(subset_perf[:, sbs_alg_idx]))

        df_sbs = pd.DataFrame(sbs_scores, columns=group_names)
        df_sbs.insert(0, "Dim", all_dims)
        df_vbs = pd.DataFrame(vbs_scores, columns=group_names)
        df_vbs.insert(0, "Dim", all_dims)
        return df_sbs.round(3), df_vbs.round(3)

    def _avg_over_folds(dict_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        fold_ids = list(dict_df.keys())
        stacked = np.stack([dict_df[f].loc[:, group_names].to_numpy(dtype=float) for f in fold_ids], axis=0)
        # nanmean prevents folds that lack a (dim,group) cell from diluting averages.
        mean = np.nanmean(stacked, axis=0)
        out = pd.DataFrame(mean, columns=group_names)
        out.insert(0, "Dim", all_dims)
        out = out.set_index("Dim").T.reset_index().rename(columns={"index": "Problem Group"})
        out.columns.name = "Problem Group"
        
        # print('dict_df: ', dict_df)
        # print('stacked: ', stacked)
        # print('mean: ', mean)
        # print('out: ', out)
        return out

    def _compute_concat_scores_df(
        values_all: np.ndarray,
        *,
        use_median: bool = False,
        quantile: Optional[float] = None,
    ) -> pd.DataFrame:
        scores = np.zeros((len(all_dims), len(group_names)), dtype=float)
        for di, dval in enumerate(all_dims):
            dim_mask = np.ones(len(preds_all), dtype=bool) if dval == "all" else (dims_all_arr == int(dval))
            for gi, gname in enumerate(group_names):
                grp_mask = np.ones(len(preds_all), dtype=bool) if gname == "all" else (group_all == gname)
                m = dim_mask & grp_mask
                if not np.any(m):
                    continue
                if quantile is not None:
                    score = float(np.quantile(values_all[m], quantile))
                elif use_median:
                    score = float(np.median(values_all[m]))
                else:
                    score = float(np.mean(values_all[m]))
                scores[di, gi] = score

        df_scores_concat = pd.DataFrame(scores, columns=group_names)
        df_scores_concat.insert(0, "Dim", all_dims)
        return _avg_over_folds({"concat": df_scores_concat.round(3)})

    df_avg_scores = _avg_over_folds(dict_df_scores)
    df_avg_accuracies = _avg_over_folds(dict_df_accuracies)
    df_avg_f1 = _avg_over_folds(dict_df_f1)
    df_avg_sbs = _avg_over_folds(dict_df_sbs)
    # Keep VBS protocol-invariant; relERT VBS is 1.0 row-wise.
    _, df_vbs_full = _compute_sbs_vbs_on_full_df(use_median=False)
    df_avg_vbs = _avg_over_folds({"full": df_vbs_full})

    preds_all = pd.concat([p.copy() for p in dict_df_predictions.values()], ignore_index=True).copy()
    preds_all.attrs = {}
    preds_all["Problem"] = preds_all["Problem"].astype(str).str.lower()
    preds_all["Dim"] = preds_all["Dim"].astype(int)

    pred_alg_all = preds_all[alg_cols].idxmin(axis=1).to_numpy()
    all_keys = pd.MultiIndex.from_frame(preds_all[["Problem", "Dim"]])
    perf_rows_all = perf_by_key.loc[all_keys].to_numpy(dtype=float)
    pred_idx_all = pd.Series(pred_alg_all).map(col_to_idx).to_numpy(dtype=int)
    picked_score_all = perf_rows_all[np.arange(len(perf_rows_all)), pred_idx_all]
    sbs_score_all = perf_rows_all[:, int(sbs_alg_idx_global)]

    fid_all = preds_all["Problem"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    group_all = np.full(fid_all.shape, "all", dtype=object)
    for name, lo, hi in group_defs:
        group_all[(fid_all >= lo) & (fid_all <= hi)] = name

    dims_all_arr = preds_all["Dim"].to_numpy()

    # Recompute gap closure using the averaged scores table and concatenated baselines.
    def _gap_from_tables(df_scores: pd.DataFrame, df_sbs: pd.DataFrame, df_vbs: pd.DataFrame) -> pd.DataFrame:
        s = df_scores.set_index("Problem Group")
        sbs = df_sbs.set_index("Problem Group")
        vbs = df_vbs.set_index("Problem Group")

        s = s.astype(float)
        sbs = sbs.astype(float)
        vbs = vbs.astype(float)

        denom = (sbs - vbs).to_numpy(dtype=float)
        num = (sbs - s).to_numpy(dtype=float)
        out = np.zeros_like(num, dtype=float)

        small = np.abs(denom) <= 1e-12
        out[~small] = num[~small] / denom[~small]
        # If SBS == VBS, define closure as 1 iff score==VBS else 0.
        out[small] = (np.abs(s.to_numpy(dtype=float)[small] - vbs.to_numpy(dtype=float)[small]) <= 1e-12).astype(float)

        df_gap = pd.DataFrame(out, index=s.index, columns=s.columns).reset_index()
        return df_gap.rename(columns={"index": "Problem Group"}).round(3)

    df_avg_gap_closure = _gap_from_tables(df_avg_scores, df_avg_sbs, df_avg_vbs)

    # Median score on the concatenation (treat all held-out cases together).
    df_median_scores = _compute_concat_scores_df(picked_score_all, use_median=True)

    # Keep the global SBS identity, but evaluate it on the duplicated held-out population.
    df_median_sbs = _compute_concat_scores_df(sbs_score_all, use_median=True)
    _, df_median_vbs_full = _compute_sbs_vbs_on_full_df(use_median=True)
    df_median_vbs = _avg_over_folds({"full": df_median_vbs_full})
    df_median_gap_closure = _gap_from_tables(df_median_scores, df_median_sbs, df_median_vbs)

    # 90th percentile (p90) score on the concatenation (treat all held-out cases together).
    df_p90_scores = _compute_concat_scores_df(picked_score_all, quantile=0.9)

    df_p90_sbs = _compute_concat_scores_df(sbs_score_all, quantile=0.9)
    _, df_p90_vbs_full = _compute_sbs_vbs_on_full_df(quantile=0.9)
    df_p90_vbs = _avg_over_folds({"full": df_p90_vbs_full})
    df_p90_gap_closure = _gap_from_tables(df_p90_scores, df_p90_sbs, df_p90_vbs)

    pick_rate = preds_all[alg_cols].idxmin(axis=1).value_counts(normalize=True).sort_values(ascending=False)
    vbs_pick_rate = true_best_alg.loc[all_keys].value_counts(normalize=True).sort_values(ascending=False)

    return {
        "scores": df_avg_scores,
        "median_scores": df_median_scores,
        "p90_scores": df_p90_scores,
        "accuracies": df_avg_accuracies,
        "accuracies_by_fold": dict_df_accuracies,
        "f1": df_avg_f1,
        "pick_rate": pick_rate,
        "vbs_pick_rate": vbs_pick_rate,
        "sbs": df_avg_sbs,
        "vbs": df_avg_vbs,
        "gap_closure": df_avg_gap_closure,
        "median_sbs": df_median_sbs,
        "median_vbs": df_median_vbs,
        "median_gap_closure": df_median_gap_closure,
        "p90_sbs": df_p90_sbs,
        "p90_vbs": df_p90_vbs,
        "p90_gap_closure": df_p90_gap_closure,
        "preds_all": preds_all,
    }
