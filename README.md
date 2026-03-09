# AS_BBOB_SOO

Train and evaluate an algorithm-selection model on BBOB-style data with three cross-validation protocols:

- **LPO**: leave-problem-out
- **LIO**: leave-instance-out
- **Random**: random split over **cases** (problem × dim × instance × repetition)

The main entrypoint is a multi-GPU parallel runner that schedules CV folds/splits across available GPUs and writes:

- A **summary CSV** (mean + median metrics)
- A **per-sample predictions table** with `Problem, Dim, Instance, Repetition, <model predictions...>`

## Repository layout

- `functions/model.py`: model definition
- `functions/model_interface.py`: dataset + training loop + CV protocols + metric/reporting utilities
- `train_parallel.py`: multi-GPU CV orchestrator (random/LPO/LIO)
- `train.sh`: minimal convenience wrapper for `train_parallel.py`
- `data/`: example CSVs and (optionally) BBOB `.npz` data

## Environment

This project is intended to run in the `as_bbo` conda environment.

```bash
conda env create -f environment.yaml
conda activate as_bbo
```

## Data expectations

`train_parallel.py` consumes a CSV with at least:

- Column 0–1: metadata columns (the code expects a `Problem` column; typical files also include `Dim`)
- Remaining columns: algorithm performance targets (one column per algorithm)

The corresponding `.npz` files are expected under `--data-root` (see `functions/model_interface.py` for the exact naming/lookup conventions).

## Usage

### Quick run (via bash wrapper)

Edit variables at the top of `train.sh` (protocol, CSV, data root, etc.), then:

```bash
bash train.sh
```

You can also override parameters via environment variables by editing the script (it is intentionally minimal).

### Run directly (recommended)

```bash
python train_parallel.py \
  --protocol all \
  --csv data/log_relert_bbob.csv \
  --data-root data/bbob \
  --resolution 16 \
  --k-views 16 \
  --num-repetitions 10 \
  --batch-size 32 \
  --num-epochs 50 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --seed 42 \
  --test-ratio 0.2 \
  --n-splits 5 \
  --gpus auto
```

Notes:
- `--protocol`: `random`, `lpo`, `lio`, or `all`.
- `--gpus auto` uses `nvidia-smi` if available; otherwise falls back to `CUDA_VISIBLE_DEVICES` or GPU `0`.
- `--max-parallel` limits concurrent trainings (default: number of selected GPUs).

### Dry run (task enumeration only)

```bash
python train_parallel.py --dry-run --protocol all --gpus 0
```

## Outputs

Results are written per protocol under:

- `results/bbob/<PROTOCOL>/res_<resolution>_k_views_<k>.csv`
- `results/bbob/<PROTOCOL>/preds_<resolution>_k_views_<k>.csv.gz`

Where `<PROTOCOL>` is one of:

- `LPO`
- `LIO`
- `RANDOM`

The summary CSV is structured as:

- Mean: `AS`, `VBS`, `SBS`, `Gap_Closure`
- Median: `AS`, `VBS`, `SBS`, `Gap_Closure`
- Then: `Accuracies`, `F1`, `Pick_Rate`, `VBS_Pick_Rate`

## TensorBoard

TensorBoard support is implemented in the training loop in `functions/model_interface.py` (via `torch.utils.tensorboard`).

If you want TensorBoard runs for CV tasks, add TB arguments where `single_train(...)` is called (the hooks are already present).

## License

Add a license if/when you publish this repository.
