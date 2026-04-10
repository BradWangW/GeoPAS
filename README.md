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

## Pipeline

1. Build a reference relERT table for the candidate algorithms.
2. Generate multi-view slice data for each BBOB problem, dimension, instance, and repetition.
3. Train and evaluate GeoPAS with one of the supported protocols: `random`, `lpo`, or `lio`.
4. Collect the outputs written by `train_parallel.py`:
	 - `res_*.csv`: summary tables for AS, SBS, VBS, gap closure, accuracies, F1, and pick rates.
	 - `preds_*.csv.gz`: per-sample predicted scores and realised selections.
5. Aggregate results across parameter settings with `concatenate_over_parameters.ipynb`.
6. Inspect budget sensitivity with `robustness_over_budget.ipynb`.
7. Run `analysis.ipynb` for validation and failure analysis.

## Data Expectations

- Training expects a relERT CSV indexed by `Problem` and `Dim`.
- Training data are stored under a root of the form `.../res_<resolution>/` and loaded from `.npz` files named like `f{fid}_i{instance}_dim{dim}_rep{rep}.npz`.
- The default shell wrapper assumes the larger project layout under `/data1/home/jw1017/AS_BBO_REBUILT/`; adjust those paths before reusing the workspace elsewhere.

## Running

Create the environment:

```bash
conda env create -f environment.yaml
conda activate as_bbo
```

Run the configured sweep:

```bash
bash train.sh
```

Or run the orchestrator directly with explicit paths:

```bash
python train_parallel.py \
	--protocol all \
	--csv /path/to/relert.csv \
	--data-root /path/to/data_root \
	--out-dir /path/to/results
```

## Notes

- `train.sh` is a local experiment launcher, not a portable interface; it currently contains machine-specific absolute paths.
- The notebooks are analysis and aggregation utilities layered on top of the CSV outputs written by the training pipeline.
