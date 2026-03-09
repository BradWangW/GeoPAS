import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import qmc

warnings.filterwarnings(
    "ignore",
    message="The balance properties of Sobol' points require n to be a power of 2",
    category=UserWarning,
    module="scipy.stats._qmc",
)


# ----------------------------- Sampling ------------------------------------ #

@dataclass(frozen=True)
class SliceBatch:
    X: np.ndarray      # (k, d, r, r) float32, clipped coords in physical bounds
    mask: np.ndarray   # (k, r, r) uint8, 1=valid (inside [0,1]^d before clipping)
    ell: np.ndarray    # (k,) float32, raw side length in [ell_min, ell_max]
    centers: Optional[np.ndarray] = None  # (k, d) float32, Sobol/Uniform centres in [0,1]^d
    B: Optional[np.ndarray] = None        # (k, 2, d) float32, 2D orthonormal frame in R^d


def _haar_2d_frames(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Approx Haar-uniform 2D frames: sample Gaussian d×2 and QR-orthonormalise."""
    A = rng.normal(size=(k, d, 2)).astype(np.float64, copy=False)
    Q = np.empty_like(A)
    for i in range(k):
        qi, _ = np.linalg.qr(A[i], mode="reduced")  # (d,2)
        Q[i] = qi
    return np.transpose(Q, (0, 2, 1)).astype(np.float32)  # (k,2,d)


def sample_random_slices(
    *,
    d: int,
    k_views: int,
    r: int,
    bounds: np.ndarray,
    ell_min: float = 0.02,
    ell_max: float = 0.5,
    log_uniform_scale: bool = True,
    use_sobol: bool = True,
    seed: Optional[int] = None,
) -> SliceBatch:
    """
    Sample k random oriented 2D square slices in [0,1]^d with:
      - centres c ~ Sobol or Uniform in [0,1]^d
      - scales ell ~ log-uniform in [ell_min, ell_max] (Sobol/Uniform in u, then map)
      - orientations ~ Haar-ish (Gaussian->QR)
    Returns coordinates clipped to [0,1]^d for safe evaluation, plus a validity mask.
    """
    if d < 2:
        raise ValueError("d must be >= 2.")
    bounds = np.asarray(bounds, dtype=np.float32)
    if bounds.shape != (d, 2):
        raise ValueError(f"bounds must have shape (d,2), got {bounds.shape}.")
    lows, highs = bounds[:, 0], bounds[:, 1]
    widths = highs - lows
    if np.any(widths <= 0):
        raise ValueError("Each bound must satisfy upper > lower.")

    rng = np.random.default_rng(seed)

    # centres + log-scale via Sobol (recommended) or RNG
    if use_sobol:
        U = qmc.Sobol(d=d + 1, scramble=True, seed=seed).random(k_views).astype(np.float32, copy=False)
    else:
        U = rng.random((k_views, d + 1), dtype=np.float32)

    centers = U[:, :d]  # (k,d) in [0,1]
    u = U[:, d]         # (k,)  in [0,1]
    if log_uniform_scale:
        ell = (ell_min * (ell_max / ell_min) ** u).astype(np.float32)  # log-uniform
    else:
        ell = ((ell_max - ell_min) * u + ell_min).astype(np.float32)  # uniform

    # orientations (k,2,d)
    B = _haar_2d_frames(d, k_views, rng)  # float32

    # r×r grid in local coords [-1/2, 1/2]^2
    s = np.linspace(-0.5, 0.5, r, dtype=np.float32)
    S, T = np.meshgrid(s, s, indexing="ij")
    grid2 = np.stack([S, T], axis=-1)  # (r,r,2)

    # Offsets: O[k,r,r,d] = grid2 @ B
    O = np.einsum("pqb,kbd->kpqd", grid2, B, dtype=np.float32)  # (k,r,r,d)

    # Construct points (may go out of [0,1])
    X_norm = centers[:, None, None, :] + ell[:, None, None, None] * O  # (k,r,r,d)

    # Validity mask BEFORE clipping
    mask = np.all((X_norm >= 0.0) & (X_norm <= 1.0), axis=-1).astype(np.uint8)  # (k,r,r)

    # Clip only to keep evaluation safe
    X_norm = np.clip(X_norm, 0.0, 1.0)

    # Rescale to physical bounds and reorder to (k,d,r,r)
    X_phys = (lows[None, None, None, :] + X_norm * widths[None, None, None, :]).astype(np.float32)
    X = np.transpose(X_phys, (0, 3, 1, 2))  # (k,d,r,r)

    return SliceBatch(X=X, mask=mask, ell=ell, centers=centers.astype(np.float32, copy=False), B=B)


# ----------------------------- Evaluation ---------------------------------- #

def evaluate_grid(
    f: Callable[[np.ndarray], float],
    points_grid: np.ndarray,   # (r,r,d)
    batch_size: int = 1000,
) -> np.ndarray:
    """Evaluate black-box f on a (r,r,d) grid, batched."""
    P = points_grid.reshape(-1, points_grid.shape[-1])
    out = np.empty((P.shape[0],), dtype=np.float32)
    for a in range(0, P.shape[0], batch_size):
        b = min(a + batch_size, P.shape[0])
        out[a:b] = np.array([f(x) for x in P[a:b]], dtype=np.float32)
    r = points_grid.shape[0]
    return out.reshape(r, r)


def normalise_plots_masked_inplace(plots: np.ndarray, mask: np.ndarray) -> None:
    """
    plots: (k,r,r) float32
    mask:  (k,r,r) uint8
    Normalise each plot to [0,1] using only valid pixels; set invalid pixels to 0.5.
    """
    k = plots.shape[0]
    valid = (mask.astype(bool)) & np.isfinite(plots)

    # initialise all invalid to 0.5
    plots[~valid] = 0.5

    for i in range(k):
        vi = valid[i]
        if not np.any(vi):
            plots[i, :, :] = 0.5
            continue
        vals = plots[i][vi]
        mn, mx = float(vals.min()), float(vals.max())
        span = mx - mn
        if not np.isfinite(span) or span < 1e-6:
            plots[i, :, :] = 0.5
            continue
        plots[i][vi] = (plots[i][vi] - mn) / span
        plots[i][~vi] = 0.5


# ----------------------------- End-to-end ---------------------------------- #

def generate_instance_views(
    f: Callable[[np.ndarray], float],
    *,
    d: int,
    k_views: int,
    r: int,
    bounds: np.ndarray,
    ell_min: float = 0.02,
    ell_max: float = 0.5,
    use_sobol: bool = True,
    seed: Optional[int] = None,
    eval_batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      plots: (k,r,r) float32 normalised
            mask : (k,r,r) uint8
            stats: (k,3)   float32, columns = [ell, range, iqr] from pre-normalised values
    """
    batch = sample_random_slices(
        d=d, k_views=k_views, r=r, bounds=bounds,
        ell_min=ell_min, ell_max=ell_max, use_sobol=use_sobol, seed=seed
    )

    plots = np.empty((k_views, r, r), dtype=np.float32)
    for i in range(k_views):
        points = batch.X[i].transpose(1, 2, 0)  # (r,r,d)
        plots[i] = evaluate_grid(f, points, batch_size=eval_batch_size)

    # --- Stats from pre-normalised values (masked) ------------------------
    valid = (batch.mask.astype(bool)) & np.isfinite(plots)
    stats = np.zeros((k_views, 3), dtype=np.float32)
    stats[:, 0] = batch.ell.astype(np.float32, copy=False)
    for i in range(k_views):
        vi = valid[i]
        if not np.any(vi):
            continue
        vals = plots[i][vi]
        mn, mx = float(vals.min()), float(vals.max())
        stats[i, 1] = np.float32(mx - mn)
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        stats[i, 2] = np.float32(q3 - q1)

    normalise_plots_masked_inplace(plots, batch.mask)
    return plots, batch.mask, stats


def save_views(
    out_dir: str,
    tag: str,
    plots: np.ndarray,
    mask: np.ndarray,
    ell: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{tag}_plots.npy"), plots.astype(np.float32))
    np.save(os.path.join(out_dir, f"{tag}_mask.npy"),  mask.astype(np.uint8))
    np.save(os.path.join(out_dir, f"{tag}_ell.npy"),   ell.astype(np.float32))


# ----------------------------- Example hook -------------------------------- #
import cocoex

def process_problem_chunk(
    func_id,
    instance_id,
    dim,
    num_repetitions,
    k_views,
    resolution,
    ell_min,
    ell_max,
    log_uniform_scale,
    out_dir,
    func_to_group,
    *,
    use_sobol=True,
    eval_batch_size=2000,
):
    """
    Generate plot views for one (function, instance, dimension).
    Saves one .npz per repetition: plots, masks, stats.
    """
    suite = cocoex.Suite(
        "bbob",
        "",
        f"dimensions:{dim} function_indices:{func_id} instance_indices:{instance_id}",
    )
    problem = suite.get_problem(0)

    bounds = np.array([[-5.0, 5.0]] * dim, dtype=np.float32)
    timing_data = []

    for rep in range(num_repetitions):
        t0 = time.time()

        # --- Sample slices ---------------------------------------------------
        batch = sample_random_slices(
            d=dim,
            k_views=k_views,
            r=resolution,
            bounds=bounds,
            ell_min=ell_min,
            ell_max=ell_max,
            log_uniform_scale=log_uniform_scale,
            use_sobol=use_sobol
        )

        # batch.X: (k, d, r, r)
        # Flatten all points at once
        points = batch.X.transpose(0, 2, 3, 1).reshape(-1, dim)

        # --- Evaluate --------------------------------------------------------
        values = np.empty((points.shape[0],), dtype=np.float32)
        for a in range(0, len(points), eval_batch_size):
            b = min(a + eval_batch_size, len(points))
            values[a:b] = np.array(
                [problem(x) for x in points[a:b]], dtype=np.float32
            )

        # Reshape back to (k, r, r)
        plots = values.reshape(k_views, resolution, resolution)

        # --- Stats from pre-normalised values (masked) ---------------------
        valid = (batch.mask.astype(bool)) & np.isfinite(plots)
        stats = np.zeros((k_views, 3), dtype=np.float32)
        stats[:, 0] = batch.ell.astype(np.float32, copy=False)
        for i in range(k_views):
            vi = valid[i]
            if not np.any(vi):
                continue
            vals = plots[i][vi]
            mn, mx = float(vals.min()), float(vals.max())
            stats[i, 1] = np.float32(mx - mn)
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            stats[i, 2] = np.float32(q3 - q1)

        # --- Mask-aware normalisation ---------------------------------------
        normalise_plots_masked_inplace(plots, batch.mask)

        # --- Save ------------------------------------------------------------
        tag = f"f{func_id}_i{instance_id}_dim{dim}_rep{rep}"
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            plots=plots.astype(np.float32),
            masks=batch.mask.astype(np.uint8),
            stats=stats.astype(np.float32),
        )

        # --- Timing bookkeeping ---------------------------------------------
        t1 = time.time()
        timing_data.append(((int(func_id), int(dim)), float(t1 - t0)))

    return timing_data
