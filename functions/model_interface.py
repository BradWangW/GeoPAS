import os
import re
import copy
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
      target (M,)    float32  (algorithm performance vector)
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
        catastrophe_log_threshold: float = np.log(36690),
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
        self.target_scale = str(target_scale).lower().strip()
        self.catastrophe_log_threshold = float(catastrophe_log_threshold)

        if self.target_scale not in {"log", "raw"}:
            raise ValueError(f"target_scale must be 'log' or 'raw', got: {target_scale!r}")

        # targets are all columns after Problem, Dim
        self.alg_cols = list(self.df.columns[2:])

        # Build index: one file per (row in df) × instance × repetition
        self.files: List[str] = []
        self.meta: List[SampleMeta] = []
        self.targets: List[torch.Tensor] = []
        self.cat_labels: List[torch.Tensor] = []

        # Optional in-memory cache
        self._cache_plots: List[torch.Tensor] = []
        self._cache_masks: List[torch.Tensor] = []
        self._cache_stats: List[torch.Tensor] = []
        self._cache_dims: List[int] = []

        def target_transform_np(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            if self.target_scale == "log":
                return np.log(np.maximum(x, 1e-6)).astype(np.float32)
            # raw scale
            return x.astype(np.float32)

        for row in range(len(self.df)):
            fid = problem_to_fid(self.df.loc[row, "Problem"])
            dim = int(self.df.loc[row, "Dim"])

            target_raw = self.df.loc[row, self.alg_cols].astype(np.float32).to_numpy()
            target_np = target_transform_np(target_raw)
            target = torch.from_numpy(target_np)
            # Catastrophe labels are defined in log(relERT) space for consistency across target scales.
            target_log = np.log(np.maximum(target_raw, 1e-6)).astype(np.float32)
            cat_np = (target_log >= self.catastrophe_log_threshold).astype(np.float32)
            cat = torch.from_numpy(cat_np)

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
                    self.cat_labels.append(cat)

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
        cat = self.cat_labels[idx]
        return plots, masks, stats, dim, target, cat

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

        self._cache_plots: List[torch.Tensor] = []
        self._cache_masks: List[torch.Tensor] = []
        self._cache_stats: List[torch.Tensor] = []
        self._cache_dims: List[int] = []
        self._cache_targets: List[torch.Tensor] = []
        self._cache_cat: List[torch.Tensor] = []

        if self.cache:
            for base_idx in self.indices:
                plots, masks, stats, dim, target, cat = self.base[base_idx]
                self._cache_plots.append(plots)
                self._cache_masks.append(masks)
                self._cache_stats.append(stats)
                self._cache_dims.append(int(dim))
                self._cache_targets.append(target)
                self._cache_cat.append(cat)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.cache:
            return (
                self._cache_plots[idx],
                self._cache_masks[idx],
                self._cache_stats[idx],
                self._cache_dims[idx],
                self._cache_targets[idx],
                self._cache_cat[idx],
            )
        base_idx = self.indices[idx]
        return self.base[base_idx]

    @property
    def problem_ids(self) -> List[str]:
        return [f"f{self.base.meta[i].fid}" for i in self.indices]

    @property
    def dims(self) -> List[int]:
        return [int(self.base.meta[i].dim) for i in self.indices]

    @property
    def instance_ids(self) -> List[int]:
        return [int(self.base.meta[i].instance) for i in self.indices]

    @property
    def repetitions(self) -> List[int]:
        return [int(self.base.meta[i].rep) for i in self.indices]


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


def _cat_accuracy_table(
    *,
    problem_ids: Sequence[str],
    dims: Sequence[int],
    cat_prob: np.ndarray,
    cat_true: np.ndarray,
    cat_tau: float,
) -> pd.DataFrame:
    """Elementwise catastrophe recognition accuracy per BBOB-group × dimension.

    Computes accuracy over all (sample, algorithm) elements in each subset.
    """
    problem_ids_s = pd.Series(problem_ids, dtype=str).str.lower()
    dims_arr = np.asarray(dims, dtype=int)
    cat_prob = np.asarray(cat_prob, dtype=float)
    cat_true = np.asarray(cat_true, dtype=float)
    if cat_prob.shape != cat_true.shape:
        raise ValueError(f"cat_prob/cat_true shape mismatch: {cat_prob.shape} vs {cat_true.shape}")
    if len(problem_ids_s) != cat_prob.shape[0] or len(dims_arr) != cat_prob.shape[0]:
        raise ValueError("problem_ids/dims must align with cat arrays")

    group_defs = [
        ("f1-f5", 1, 5),
        ("f6-f9", 6, 9),
        ("f10-f14", 10, 14),
        ("f15-f19", 15, 19),
        ("f20-f24", 20, 24),
    ]
    group_names = [g for (g, _, _) in group_defs] + ["all"]

    fid = problem_ids_s.str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    group = np.full(fid.shape, "all", dtype=object)
    for name, lo, hi in group_defs:
        group[(fid >= lo) & (fid <= hi)] = name

    dims_list = sorted(pd.unique(dims_arr))
    all_dims: List[object] = list(dims_list) + ["all"]

    cat_pred = (cat_prob >= float(cat_tau))
    cat_true_b = (cat_true >= 0.5)
    # Use NaN for subsets with no samples so fold-averaging does not dilute values
    # (important for LPO where each fold covers only one problem group).
    acc = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)

    for di, dval in enumerate(all_dims):
        dim_mask = np.ones(len(dims_arr), dtype=bool) if dval == "all" else (dims_arr == int(dval))
        for gi, gname in enumerate(group_names):
            grp_mask = np.ones(len(dims_arr), dtype=bool) if gname == "all" else (group == gname)
            m = dim_mask & grp_mask
            if not np.any(m):
                continue
            acc[di, gi] = float(np.mean(cat_pred[m] == cat_true_b[m]))

    df = pd.DataFrame(acc, columns=group_names)
    df.insert(0, "Dim", all_dims)
    return df


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
    cat_loss_weight: float = 15.0,
    cat_tau: float = 0.5,
    cat_penalty: float = 15,
    return_cat_arrays: bool = False,
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
    # reg_loss_fn = nn.SmoothL1Loss(beta=12.477)
    cat_loss_fn = nn.BCEWithLogitsLoss()

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
                raise ValueError(f"Model must return (pred, cat_logits|None), got {type(out)} of len {len(out)}")
            pred, cat_logits = out
            return pred, cat_logits
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
            "train/loss_reg": [],
            "train/loss_cat": [],
            "train/as": [],
            "train/lr": [],
            "val/loss": [],
            "val/loss_reg": [],
            "val/loss_cat": [],
            "val/as": [],
        }

        best_val: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        for epoch in pbar:
            epoch_losses = []
            epoch_reg_losses = []
            epoch_cat_losses = []
            epoch_train_as_total = 0.0
            epoch_train_as_count = 0
            for step, (plots, masks, stats, dim, target, cat) in enumerate(train_es_loader):
                plots = plots.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                dim = dim.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                cat = cat.to(device, non_blocking=True)
                
                optim.zero_grad(set_to_none=True)
                pred, cat_logits = _forward(plots, masks, stats, dim)

                loss_reg = reg_loss_fn(pred, target)
                use_cat = (cat_logits is not None) and (cat_loss_weight != 0.0)
                loss_cat = cat_loss_fn(cat_logits, cat) if use_cat else pred.new_zeros(())
                loss = loss_reg + float(cat_loss_weight) * loss_cat

                if cat_logits is not None:
                    cat_prob = torch.sigmoid(cat_logits)
                    unsafe = cat_prob >= float(cat_tau)
                    train_scores = pred + float(cat_penalty) * unsafe.to(dtype=pred.dtype)
                else:
                    train_scores = pred
                train_pick = torch.argmin(train_scores, dim=1)
                train_achieved = target.gather(1, train_pick.unsqueeze(1)).squeeze(1)
                epoch_train_as_total += float(train_achieved.sum().item())
                epoch_train_as_count += int(train_achieved.numel())
                # loss = reg_loss_fn(pred, target) + 2 * _pairwise_logistic_ranking_loss(
                #     pred, target
                # )
                # loss = asymmetric_mse(pred, target, alpha=10.0)
                # loss = percentile_weighted_loss(pred, target)
                
                loss.backward()
                optim.step()

                epoch_losses.append(loss.item())
                epoch_reg_losses.append(float(loss_reg.detach().item()))
                epoch_cat_losses.append(float(cat_loss_weight) * float(loss_cat.detach().item()))

            # Validation score for early stopping (mean objective, lower is better).
            val_score: Optional[float] = None
            val_loss: Optional[float] = None
            val_loss_reg: Optional[float] = None
            val_loss_cat: Optional[float] = None
            if val_loader is not None:
                model.eval()
                total = 0
                total_as = 0.0
                total_val_loss = 0.0
                total_val_loss_reg = 0.0
                total_val_loss_cat = 0.0
                with torch.no_grad():
                    for plots, masks, stats, dim, target, cat in val_loader:
                        plots = plots.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        stats = stats.to(device, non_blocking=True)
                        dim = dim.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        cat = cat.to(device, non_blocking=True)
                        pred, cat_logits = _forward(plots, masks, stats, dim)

                        v_loss_reg = reg_loss_fn(pred, target)
                        v_use_cat = (cat_logits is not None) and (cat_loss_weight != 0.0)
                        v_loss_cat = cat_loss_fn(cat_logits, cat) if v_use_cat else pred.new_zeros(())
                        v_loss = v_loss_reg + float(cat_loss_weight) * v_loss_cat

                        # Downstream validation mean AS (lower is better):
                        # pick algorithm with the lowest predicted (penalized) score, then
                        # evaluate by the achieved true target value.
                        if cat_logits is not None:
                            cat_prob = torch.sigmoid(cat_logits)
                            unsafe = cat_prob >= float(cat_tau)
                            scores = pred + float(cat_penalty) * unsafe.to(dtype=pred.dtype)
                        else:
                            scores = pred

                        pick = torch.argmin(scores, dim=1)  # (B,)
                        achieved = target.gather(1, pick.unsqueeze(1)).squeeze(1)  # (B,)
                        bs = int(achieved.numel())
                        total_as += float(achieved.sum().item())
                        total_val_loss += float(v_loss.item()) * bs
                        total_val_loss_reg += float(v_loss_reg.item()) * bs
                        total_val_loss_cat += (float(cat_loss_weight) * float(v_loss_cat.item())) * bs
                        total += bs

                if total > 0:
                    val_score = float(total_as / total)
                    val_loss = float(total_val_loss / total)
                    val_loss_reg = float(total_val_loss_reg / total)
                    val_loss_cat = float(total_val_loss_cat / total)
                    if writer is not None and tb_log_val:
                        writer.add_scalar("val/as", val_score, epoch)
                        writer.add_scalar("val/loss", val_loss, epoch)
                        writer.add_scalar("val/loss_reg", val_loss_reg, epoch)
                        writer.add_scalar("val/loss_cat", val_loss_cat, epoch)
                model.train()

            # update the epoch progress bar
            if epoch_losses:
                avg_loss = float(sum(epoch_losses) / len(epoch_losses))
                avg_reg = float(sum(epoch_reg_losses) / len(epoch_reg_losses)) if epoch_reg_losses else 0.0
                avg_cat = float(sum(epoch_cat_losses) / len(epoch_cat_losses)) if epoch_cat_losses else 0.0
                train_as = float(epoch_train_as_total / epoch_train_as_count) if epoch_train_as_count > 0 else float("nan")
                if val_score is None:
                    pbar.set_postfix(reg=f"{avg_reg:.6f}", cat=f"{avg_cat:.6f}", train_as=f"{train_as:.6f}")
                else:
                    pbar.set_postfix(reg=f"{avg_reg:.6f}", cat=f"{avg_cat:.6f}", train_as=f"{train_as:.6f}", val=f"{val_score:.6f}")
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, epoch)
                    writer.add_scalar("train/loss_reg", avg_reg, epoch)
                    writer.add_scalar("train/loss_cat", avg_cat, epoch)
                    writer.add_scalar("train/as", train_as, epoch)
                    writer.add_scalar("train/lr", float(optim.param_groups[0]["lr"]), epoch)

                history["train/loss"].append(float(avg_loss))
                history["train/loss_reg"].append(float(avg_reg))
                history["train/loss_cat"].append(float(avg_cat))
                history["train/as"].append(float(train_as))
                history["train/lr"].append(float(optim.param_groups[0]["lr"]))
                history["val/loss"].append(float(val_loss) if val_loss is not None else float("nan"))
                history["val/loss_reg"].append(float(val_loss_reg) if val_loss_reg is not None else float("nan"))
                history["val/loss_cat"].append(float(val_loss_cat) if val_loss_cat is not None else float("nan"))
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
    cat_true_all = []
    cat_prob_all = []
    with torch.no_grad():
        for plots, masks, stats, dim, _, cat in test_loader:
            plots = plots.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            dim   = dim.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)

            pred, cat_logits = _forward(plots, masks, stats, dim)
            if cat_logits is not None:
                cat_prob = torch.sigmoid(cat_logits)
                unsafe = cat_prob >= float(cat_tau)
                scores = pred + float(cat_penalty) * unsafe.to(dtype=pred.dtype)
                cat_true_all.append(cat.detach().cpu())
                cat_prob_all.append(cat_prob.detach().cpu())
            else:
                scores = pred
            preds_all.append(scores.detach().cpu().numpy())

    # Catastrophe head metrics (elementwise across algorithms)
    if cat_true_all and cat_prob_all:
        cat_true_np = torch.cat(cat_true_all, dim=0).numpy()
        cat_prob_np = torch.cat(cat_prob_all, dim=0).numpy()
        # cat_pred_np = (cat_prob_np >= float(cat_tau)).astype(np.float32)

        # cat_acc = float(np.mean(cat_pred_np == cat_true_np))
        # tp = float(np.sum((cat_pred_np == 1.0) & (cat_true_np == 1.0)))
        # fp = float(np.sum((cat_pred_np == 1.0) & (cat_true_np == 0.0)))
        # fn = float(np.sum((cat_pred_np == 0.0) & (cat_true_np == 1.0)))
        # cat_prec = 0.0 if (tp + fp) <= 0.0 else tp / (tp + fp)
        # cat_rec = 0.0 if (tp + fn) <= 0.0 else tp / (tp + fn)
        # cat_f1 = 0.0 if (cat_prec + cat_rec) <= 0.0 else (2.0 * cat_prec * cat_rec) / (cat_prec + cat_rec)
        # true_rate = float(np.mean(cat_true_np))
        # pred_rate = float(np.mean(cat_pred_np))

        # print(
        #     f"[cat eval @tau={float(cat_tau):.3f}] acc={cat_acc:.4f} prec={cat_prec:.4f} rec={cat_rec:.4f} f1={cat_f1:.4f} "
        #     f"pos_rate_true={true_rate:.4f} pos_rate_pred={pred_rate:.4f}",
        #     flush=True,
        # )

        if return_cat_arrays:
            if return_history:
                return (
                    np.concatenate(preds_all, axis=0),
                    cat_prob_np,
                    cat_true_np,
                    history,
                )
            return (
                np.concatenate(preds_all, axis=0),
                cat_prob_np,
                cat_true_np,
            )

    if return_cat_arrays:
        # Fallback: should not happen in normal runs, but keep return shape stable.
        empty = np.zeros((0, 0), dtype=np.float32)
        if return_history:
            return (np.concatenate(preds_all, axis=0), empty, empty, history)
        return (np.concatenate(preds_all, axis=0), empty, empty)

    if return_history:
        return np.concatenate(preds_all, axis=0), history

    return np.concatenate(preds_all, axis=0)


def tail_table(df_perf, cap=36690, T_list=(12.477,100,300), tol=1e-3):
    alg_cols = list(df_perf.columns[2:])
    rows = {}
    for alg in alg_cols:
        v = df_perf[alg].to_numpy()
        hit_cap = (np.abs(v - cap) <= tol*cap).mean()
        row = {"P(hit_cap)": hit_cap}
        for T in T_list:
            row[f"P(relERT>{T})"] = (v > T).mean()
        rows[alg] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def compute_risk_penalty(
    df_tail: pd.DataFrame,
    *,
    cap_col: str = "P(hit_cap)",
    thr_col: Optional[str] = "P(relERT>12.477)",
    lam_cap: float = 1.0,
    lam_thr: float = 1.0,
) -> np.ndarray:
    """
    Returns penalty vector p[a] aligned to alg_cols.
    """
    alg_cols = list(df_tail.index)
    missing = [a for a in alg_cols if a not in df_tail.index]
    if missing:
        raise KeyError(f"df_tail missing algorithms: {missing}")

    if cap_col not in df_tail.columns:
        raise KeyError(f"df_tail missing column {cap_col!r}")
    if thr_col is not None and thr_col not in df_tail.columns:
        raise KeyError(f"df_tail missing column {thr_col!r}")

    p_cap = df_tail.loc[list(alg_cols), cap_col].to_numpy(dtype=np.float64)
    p_thr = np.zeros_like(p_cap)
    if thr_col is not None:
        p_thr = df_tail.loc[list(alg_cols), thr_col].to_numpy(dtype=np.float64)

    return lam_cap * p_cap + lam_thr * p_thr


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
    cat_loss_weight: float = 15.0,
    cat_tau: float = 0.5,
    cat_penalty: float = 15.0,
    penalty: Optional[np.ndarray] = None,
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
    preds, cat_prob_np, cat_true_np, tb_history = single_train(
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
        cat_loss_weight=float(cat_loss_weight),
        cat_tau=float(cat_tau),
        cat_penalty=float(cat_penalty),
        return_cat_arrays=True,
        return_history=True,
        early_stopping_patience=int(early_stopping_patience),
    )

    scores = preds
    if penalty is not None:
        scores = scores + np.asarray(penalty, dtype=np.float64)[None, :]

    out = pd.DataFrame(scores, columns=list(alg_cols))
    out.insert(0, "Repetition", test_ds.repetitions)
    out.insert(0, "Instance", test_ds.instance_ids)
    out.insert(0, "Dim", test_ds.dims)
    out.insert(0, "Problem", test_ds.problem_ids)
    out.attrs["tb_history"] = tb_history

    try:
        out.attrs["cat_accuracy"] = _cat_accuracy_table(
            problem_ids=test_ds.problem_ids,
            dims=test_ds.dims,
            cat_prob=cat_prob_np,
            cat_true=cat_true_np,
            cat_tau=float(cat_tau),
        ).round(6).to_dict(orient="split")
        out.attrs["cat_tau"] = float(cat_tau)
    except Exception:
        pass

    del model
    torch.cuda.empty_cache()
    return out


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
    cat_loss_weight: float = 15.0,
    cat_tau: float = 0.5,
    cat_penalty: float = 15.0,
    use_tail_penalty: bool = False,
    tail_lam_cap: float = 15.0,
    tail_lam_thr: float = 3.0,
    tail_scale: float = 1.0,
    pbar_head: str = "[train]",
    tb_log_dir: Optional[str] = None,
    tb_run_name: Optional[str] = None,
    tb_log_val: bool = False,
    val_ratio: float = 0.0,
    early_stopping_patience: int = 15,
) -> pd.DataFrame:
    alg_cols = list(df_train.columns[2:])

    train_ds = MultiViewNPZDataset(
        df_train,
        data_dir,
        instances=train_instances,
        num_repetitions=num_repetitions,
        cache=cache_train,
        strict=strict,
        k_views=k_views,
        target_scale=str(target_scale),
    )
    test_ds = MultiViewNPZDataset(
        df_test,
        data_dir,
        instances=test_instances,
        num_repetitions=num_repetitions,
        cache=cache_test,
        strict=strict,
        k_views=k_views,
        target_scale=str(target_scale),
    )

    penalty: Optional[np.ndarray] = None
    if use_tail_penalty:
        df_tail_train = tail_table(df_train)
        penalty = float(tail_scale) * compute_risk_penalty(
            df_tail_train,
            lam_cap=float(tail_lam_cap),
            lam_thr=float(tail_lam_thr),
        )

    return _train_predict_from_datasets(
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
        cat_loss_weight=float(cat_loss_weight),
        cat_tau=float(cat_tau),
        cat_penalty=float(cat_penalty),
        penalty=penalty,
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
            cat_loss_weight=15.0,
            cat_tau=0.5,
            cat_penalty=15.0,
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
        )
        test_ds = MultiViewNPZDataset(
            df,
            data_dir,
            instances=[test_inst],
            num_repetitions=num_repetitions,
            cache=cache_test,
            strict=strict,
            k_views=k_views,
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
            cat_loss_weight=15.0,
            cat_tau=0.5,
            cat_penalty=15.0,
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
    - log_scores: mean log(relERT) achieved by the picked algorithm
    - log_median_scores: median log(relERT) achieved by the picked algorithm (computed on concatenation of all folds)
    - log_p90_scores: 90th percentile log(relERT) achieved by the picked algorithm (computed on concatenation of all folds)
    - accuracies: exact-match accuracy vs the true best algorithm
    - f1: macro-F1 over algorithms
    - pick_rate: algorithm pick frequency (from the first fold)
    - vbs_pick_rate: oracle pick frequency (true best algorithm distribution)
    - sbs: SBS (single best solver) true relERT baseline
    - vbs: VBS (virtual best solver) true relERT baseline
    - gap_closure: VBS–SBS gap closure of the model selection
    - median_sbs: SBS baseline using median aggregation (computed on concatenation of all folds)
    - median_vbs: VBS baseline using median aggregation (computed on concatenation of all folds)
    - median_gap_closure: gap closure using median AS/SBS/VBS (computed on concatenation of all folds)
    - p90_sbs: SBS baseline using 90th percentile aggregation (computed on concatenation of all folds)
    - p90_vbs: VBS baseline using 90th percentile aggregation (computed on concatenation of all folds)
    - p90_gap_closure: gap closure using p90 AS/SBS/VBS (computed on concatenation of all folds)
    - log_sbs/log_vbs/log_gap_closure: mean log(relERT) SBS/VBS/gap-closure tables
    - log_median_sbs/log_median_vbs/log_median_gap_closure: median log(relERT) SBS/VBS/gap-closure tables
    - log_p90_sbs/log_p90_vbs/log_p90_gap_closure: p90 log(relERT) SBS/VBS/gap-closure tables
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

    def _metric_space(values: np.ndarray, *, use_log: bool) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not use_log:
            return values
        return np.log(np.maximum(values, 1e-6))

    # SBS definition (global): choose ONE algorithm over the entire dataset `df`.
    # Then report its performance on each subgroup/dimension subset.
    perf_all = perf_by_key.to_numpy(dtype=float)
    perf_all_log = _metric_space(perf_all, use_log=True)
    sbs_alg_idx_global = int(np.argmin(np.mean(perf_all, axis=0)))
    log_sbs_alg_idx_global = int(np.argmin(np.mean(perf_all_log, axis=0)))
    # SBS algorithm should be defined concistently
    sbs_alg_idx_med_global = sbs_alg_idx_global
    sbs_alg_idx_p90_global = sbs_alg_idx_global
    log_sbs_alg_idx_med_global = log_sbs_alg_idx_global
    log_sbs_alg_idx_p90_global = log_sbs_alg_idx_global
    # sbs_alg_idx_med_global = int(np.argmin(np.median(perf_all, axis=0)))
    # # p90-SBS (global): choose algorithm minimizing the 90th percentile relERT.
    # sbs_alg_idx_p90_global = int(np.argmin(np.quantile(perf_all, 0.9, axis=0)))

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
    dict_df_log_scores: Dict[str, pd.DataFrame] = {}
    dict_df_log_sbs: Dict[str, pd.DataFrame] = {}
    dict_df_log_vbs: Dict[str, pd.DataFrame] = {}
    dict_df_log_gap: Dict[str, pd.DataFrame] = {}

    # Extract catastrophe-accuracy payloads (if present) before we sanitize attrs.
    dict_df_cat_acc: Dict[str, pd.DataFrame] = {}
    for fold_id, preds in dict_df_predictions.items():
        payload = getattr(preds, "attrs", {}).get("cat_accuracy")
        if isinstance(payload, dict) and {"columns", "data"}.issubset(payload.keys()):
            try:
                dict_df_cat_acc[fold_id] = pd.DataFrame(payload["data"], columns=payload["columns"])
            except Exception:
                pass

    for fold_id, preds in dict_df_predictions.items():
        preds = preds.copy()
        preds.attrs = {}
        preds["Problem"] = preds["Problem"].astype(str).str.lower()
        preds["Dim"] = preds["Dim"].astype(int)

        pred_alg = preds[alg_cols].idxmin(axis=1).to_numpy()
        keys = pd.MultiIndex.from_frame(preds[["Problem", "Dim"]])

        # True relERT row for each sample, then pick the column of the predicted algorithm.
        perf_rows = perf_by_key.loc[keys].to_numpy(dtype=float)
        perf_rows_log = _metric_space(perf_rows, use_log=True)
        pred_idx = pd.Series(pred_alg).map(col_to_idx).to_numpy(dtype=int)
        picked_score = perf_rows[np.arange(len(perf_rows)), pred_idx]
        picked_score_log = perf_rows_log[np.arange(len(perf_rows_log)), pred_idx]

        sbs_alg_idx = sbs_alg_idx_global
        log_sbs_alg_idx = log_sbs_alg_idx_global

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
        log_performances = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        log_sbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        log_vbs_scores = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
        log_gap_closure = np.full((len(all_dims), len(group_names)), np.nan, dtype=float)
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

                subset_perf_log = perf_rows_log[m]
                as_score_log = float(np.mean(picked_score_log[m]))
                vbs_log = float(np.mean(np.min(subset_perf_log, axis=1)))
                sbs_log = float(np.mean(subset_perf_log[:, log_sbs_alg_idx]))
                log_performances[di, gi] = as_score_log
                log_vbs_scores[di, gi] = vbs_log
                log_sbs_scores[di, gi] = sbs_log

                denom_log = (sbs_log - vbs_log)
                if abs(denom_log) <= 1e-12:
                    log_gap_closure[di, gi] = 1.0 if abs(as_score_log - vbs_log) <= 1e-12 else 0.0
                else:
                    log_gap_closure[di, gi] = (sbs_log - as_score_log) / denom_log

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
        df_log_scores = pd.DataFrame(log_performances, columns=group_names)
        df_log_scores.insert(0, "Dim", all_dims)
        df_log_sbs = pd.DataFrame(log_sbs_scores, columns=group_names)
        df_log_sbs.insert(0, "Dim", all_dims)
        df_log_vbs = pd.DataFrame(log_vbs_scores, columns=group_names)
        df_log_vbs.insert(0, "Dim", all_dims)
        df_log_gap = pd.DataFrame(log_gap_closure, columns=group_names)
        df_log_gap.insert(0, "Dim", all_dims)

        # Keep the same print behaviour as before (overall == last row/col)
        print(
            "Fold-local overall for",
            fold_id,
            ", score:",
            round(float(df_scores.loc[df_scores.index[-1], "all"]), 3),
            ", SBS:",
            round(float(df_sbs.loc[df_sbs.index[-1], "all"]), 3),
            ", VBS:",
            round(float(df_vbs.loc[df_vbs.index[-1], "all"]), 3),
            ", gap_closure:",
            round(float(df_gap.loc[df_gap.index[-1], "all"]), 3),
            ", accuracy:",
            round(float(df_acc.loc[df_acc.index[-1], "all"]), 3),
            ", F1:",
            round(float(df_f1.loc[df_f1.index[-1], "all"]), 3),
        )

        dict_df_scores[fold_id] = df_scores.round(3)
        dict_df_accuracies[fold_id] = df_acc.round(3)
        dict_df_f1[fold_id] = df_f1.round(3)
        dict_df_sbs[fold_id] = df_sbs.round(3)
        dict_df_vbs[fold_id] = df_vbs.round(3)
        dict_df_gap[fold_id] = df_gap.round(3)
        dict_df_log_scores[fold_id] = df_log_scores.round(3)
        dict_df_log_sbs[fold_id] = df_log_sbs.round(3)
        dict_df_log_vbs[fold_id] = df_log_vbs.round(3)
        dict_df_log_gap[fold_id] = df_log_gap.round(3)

    def _compute_sbs_vbs_on_full_df(
        *,
        use_median: bool = False,
        quantile: Optional[float] = None,
        metric_space: str = "raw",
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
        metric_space = str(metric_space).lower().strip()
        if metric_space not in {"raw", "log"}:
            raise ValueError(f"metric_space must be 'raw' or 'log', got {metric_space!r}")
        gt = perf_by_key.reset_index().copy()  # columns: Problem, Dim, <alg...>
        gt["Problem"] = gt["Problem"].astype(str).str.lower()
        gt["Dim"] = gt["Dim"].astype(int)

        perf = gt[alg_cols].to_numpy(dtype=float)
        perf_metric = _metric_space(perf, use_log=(metric_space == "log"))

        if metric_space == "log":
            if quantile is not None:
                sbs_alg_idx = int(log_sbs_alg_idx_p90_global)
            else:
                sbs_alg_idx = int(log_sbs_alg_idx_med_global) if use_median else int(log_sbs_alg_idx_global)
        else:
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
    df_log_avg_scores = _avg_over_folds(dict_df_log_scores)
    # raise
    df_avg_accuracies = _avg_over_folds(dict_df_accuracies)
    df_avg_f1 = _avg_over_folds(dict_df_f1)

    # Catastrophe recognition accuracies (extracted earlier from attrs payloads)
    df_avg_cat_acc: Optional[pd.DataFrame] = None
    if dict_df_cat_acc:
        df_avg_cat_acc = _avg_over_folds(dict_df_cat_acc)
    # Compute SBS/VBS baselines on the full ground-truth df (protocol-invariant).
    df_sbs_full, df_vbs_full = _compute_sbs_vbs_on_full_df(use_median=False)
    df_avg_sbs = _avg_over_folds({"full": df_sbs_full})
    df_avg_vbs = _avg_over_folds({"full": df_vbs_full})
    df_log_sbs_full, df_log_vbs_full = _compute_sbs_vbs_on_full_df(use_median=False, metric_space="log")
    df_log_avg_sbs = _avg_over_folds({"full": df_log_sbs_full})
    df_log_avg_vbs = _avg_over_folds({"full": df_log_vbs_full})

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
    df_log_avg_gap_closure = _gap_from_tables(df_log_avg_scores, df_log_avg_sbs, df_log_avg_vbs)

    preds_all = pd.concat([p.copy() for p in dict_df_predictions.values()], ignore_index=True).copy()
    preds_all.attrs = {}
    preds_all["Problem"] = preds_all["Problem"].astype(str).str.lower()
    preds_all["Dim"] = preds_all["Dim"].astype(int)

    # Median score on the concatenation (treat all held-out cases together).
    pred_alg_all = preds_all[alg_cols].idxmin(axis=1).to_numpy()
    all_keys = pd.MultiIndex.from_frame(preds_all[["Problem", "Dim"]])
    perf_rows_all = perf_by_key.loc[all_keys].to_numpy(dtype=float)
    perf_rows_all_log = _metric_space(perf_rows_all, use_log=True)
    pred_idx_all = pd.Series(pred_alg_all).map(col_to_idx).to_numpy(dtype=int)
    picked_score_all = perf_rows_all[np.arange(len(perf_rows_all)), pred_idx_all]
    picked_score_all_log = perf_rows_all_log[np.arange(len(perf_rows_all_log)), pred_idx_all]

    fid_all = preds_all["Problem"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    group_all = np.full(fid_all.shape, "all", dtype=object)
    for name, lo, hi in group_defs:
        group_all[(fid_all >= lo) & (fid_all <= hi)] = name

    dims_all_arr = preds_all["Dim"].to_numpy()
    df_median_scores = _compute_concat_scores_df(picked_score_all, use_median=True)
    df_log_median_scores = _compute_concat_scores_df(picked_score_all_log, use_median=True)

    # Median baselines are computed on the full ground-truth df (protocol-invariant).
    df_median_sbs_full, df_median_vbs_full = _compute_sbs_vbs_on_full_df(use_median=True)
    df_median_sbs = _avg_over_folds({"full": df_median_sbs_full})
    df_median_vbs = _avg_over_folds({"full": df_median_vbs_full})
    df_median_gap_closure = _gap_from_tables(df_median_scores, df_median_sbs, df_median_vbs)
    df_log_median_sbs_full, df_log_median_vbs_full = _compute_sbs_vbs_on_full_df(use_median=True, metric_space="log")
    df_log_median_sbs = _avg_over_folds({"full": df_log_median_sbs_full})
    df_log_median_vbs = _avg_over_folds({"full": df_log_median_vbs_full})
    df_log_median_gap_closure = _gap_from_tables(df_log_median_scores, df_log_median_sbs, df_log_median_vbs)

    # 90th percentile (p90) score on the concatenation (treat all held-out cases together).
    df_p90_scores = _compute_concat_scores_df(picked_score_all, quantile=0.9)
    df_log_p90_scores = _compute_concat_scores_df(picked_score_all_log, quantile=0.9)

    # p90 baselines are computed on the full ground-truth df (protocol-invariant).
    df_p90_sbs_full, df_p90_vbs_full = _compute_sbs_vbs_on_full_df(quantile=0.9)
    df_p90_sbs = _avg_over_folds({"full": df_p90_sbs_full})
    df_p90_vbs = _avg_over_folds({"full": df_p90_vbs_full})
    df_p90_gap_closure = _gap_from_tables(df_p90_scores, df_p90_sbs, df_p90_vbs)
    df_log_p90_sbs_full, df_log_p90_vbs_full = _compute_sbs_vbs_on_full_df(quantile=0.9, metric_space="log")
    df_log_p90_sbs = _avg_over_folds({"full": df_log_p90_sbs_full})
    df_log_p90_vbs = _avg_over_folds({"full": df_log_p90_vbs_full})
    df_log_p90_gap_closure = _gap_from_tables(df_log_p90_scores, df_log_p90_sbs, df_log_p90_vbs)

    pick_rate = preds_all[alg_cols].idxmin(axis=1).value_counts(normalize=True).sort_values(ascending=False)
    vbs_pick_rate = true_best_alg.loc[all_keys].value_counts(normalize=True).sort_values(ascending=False)

    return {
        "scores": df_avg_scores,
        "median_scores": df_median_scores,
        "p90_scores": df_p90_scores,
        "log_scores": df_log_avg_scores,
        "log_median_scores": df_log_median_scores,
        "log_p90_scores": df_log_p90_scores,
        "accuracies": df_avg_accuracies,
        "accuracies_by_fold": dict_df_accuracies,
        "cat_accuracies": df_avg_cat_acc,
        "cat_accuracies_by_fold": dict_df_cat_acc,
        "f1": df_avg_f1,
        "pick_rate": pick_rate,
        "vbs_pick_rate": vbs_pick_rate,
        "sbs": df_avg_sbs,
        "vbs": df_avg_vbs,
        "gap_closure": df_avg_gap_closure,
        "log_sbs": df_log_avg_sbs,
        "log_vbs": df_log_avg_vbs,
        "log_gap_closure": df_log_avg_gap_closure,
        "median_sbs": df_median_sbs,
        "median_vbs": df_median_vbs,
        "median_gap_closure": df_median_gap_closure,
        "log_median_sbs": df_log_median_sbs,
        "log_median_vbs": df_log_median_vbs,
        "log_median_gap_closure": df_log_median_gap_closure,
        "p90_sbs": df_p90_sbs,
        "p90_vbs": df_p90_vbs,
        "p90_gap_closure": df_p90_gap_closure,
        "log_p90_sbs": df_log_p90_sbs,
        "log_p90_vbs": df_log_p90_vbs,
        "log_p90_gap_closure": df_log_p90_gap_closure,
        "preds_all": preds_all,
    }
