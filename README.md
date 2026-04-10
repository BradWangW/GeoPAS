# GeoPAS

This workspace contains the GeoPAS training and analysis pipeline for algorithm selection on BBOB using multi-view two-dimensional slices of black-box functions.

## Structure

- `train_parallel.py`: main training entrypoint; schedules cross-validation tasks across GPUs and writes per-protocol summary and prediction files.
- `train.sh`: local sweep wrapper around `train_parallel.py`; defines the parameter grid and output layout used in the current experiments.
- `functions/model.py`: convolutional selector model.
- `functions/model_interface.py`: dataset loading, training, evaluation, and result-table generation.
- `data_generation/plots/plot_generation_soo_extensive.py`: generates `.npz` multi-view training data from BBOB problems.
- `data_generation/performances/ERT_cal.ipynb`: computes reference performance tables used to build relERT labels.
- `analysis.ipynb`: validates result tables and performs protocol-level and cell-level failure analysis.
- `concatenate_over_parameters.ipynb`: aggregates many `res_*.csv` outputs into sectioned comparison files.
- `robustness_over_budget.ipynb`: compiles aggregated result tables into `res`, `k_views`, and budget summaries and heatmaps.

## Step-by-Step Guide

The commands below assume you are running from this workspace root.

1. Create the environment and activate it.

```bash
export PROJECT_ROOT=/path/to/AS_BBO_REBUILT
conda env create -f environment.yaml
conda activate as_bbo
```

2. Build the reference relERT table.

```bash
code data_generation/performances/ERT_cal.ipynb
```

Run all cells in the notebook. The target artifact is `data_generation/performances/relert.csv`.

3. Generate the multi-view `.npz` data.

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
python data_generation/plots/plot_generation_soo_extensive.py
```

As currently written, this script generates data under `$PROJECT_ROOT/data/bbob_by_deepela/maxscale_0.7_logscale_false/`.

4. Train and evaluate GeoPAS.

If you want to run the current configured sweep exactly as `train.sh` defines it:

```bash
bash train.sh
```

If you want to train on the data generated in step 3, point `DATA_ROOT` at that output explicitly:

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
DATA_ROOT="$PROJECT_ROOT/data/bbob_by_deepela/maxscale_0.7_logscale_false" \
bash train.sh
```

This writes per-run outputs under `$PROJECT_ROOT/results/bbob_by_deepela/results/bbob/...`, including `res_*.csv` and `preds_*.csv.gz`.

5. Aggregate result tables across parameter settings.

```bash
code concatenate_over_parameters.ipynb
```

Run the first code cell to create `AS_mean_median_p90__{LPO,LIO,RANDOM}__ALL_RUNS.csv`, then run the second code cell to create `AS_mean_median_p90__MERGED__ALL_RUNS.csv`.

6. Inspect robustness over budget.

```bash
code robustness_over_budget.ipynb
```

Run the notebook cells after setting `protocol` to the split you want to inspect.

7. Validate outputs and inspect failure modes.

```bash
code analysis.ipynb
```

Run all cells. Outputs are written under `analysis_outputs/failure_analysis/...` inside this workspace.

## Data Expectations

- The canonical relERT table used by this workspace is `data_generation/performances/relert.csv`.
- Training expects a relERT CSV indexed by `Problem` and `Dim`.
- Training data are stored under a root of the form `.../res_<resolution>/` and loaded from `.npz` files named like `f{fid}_i{instance}_dim{dim}_rep{rep}.npz`.
- By default, the shell script, notebooks, and data-generation script resolve the external `data/` and `results/` trees relative to the parent directory of this workspace, matching the current layout without changing the underlying pipeline.

If the `code` shell command is not available on your machine, open the same notebooks directly from the current VS Code workspace and run the cells there.

## Common Overrides

The main training path overrides are environment variables or shell variables:

```bash
PROJECT_ROOT=/path/to/AS_BBO_REBUILT \
RESULTS_ROOT=/path/to/results/bbob_by_deepela/results \
DATA_ROOT=/path/to/generated_npz_root \
OUT_DIR_BASE=/path/to/results/bbob \
TB_LOG_DIR=/path/to/results/tensorboard \
bash train.sh
```

The notebooks use the same default root resolution and honor `PROJECT_ROOT` and `RESULTS_ROOT` when you want to point them elsewhere.

If you want to bypass `train.sh`, run the orchestrator directly with explicit paths:

```bash
python train_parallel.py \
	--protocol all \
	--csv data_generation/performances/relert.csv \
	--data-root /path/to/data_root \
	--out-dir /path/to/results
```

## Notes

- `train.sh` preserves the current experiment behavior, but its default paths are now computed from the workspace location instead of fixed machine-specific literals.
- The notebooks are analysis and aggregation utilities layered on top of the CSV outputs written by the training pipeline.
