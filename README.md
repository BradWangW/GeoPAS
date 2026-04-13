# GeoPAS

[![arXiv](https://img.shields.io/badge/arXiv-2604.09095-b31b1b.svg)](https://arxiv.org/abs/2604.09095)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

GeoPAS is a learning-based framework for automated algorithm selection in continuous black-box optimisation. It predicts, for each problem instance, which solver in a candidate portfolio is most likely to perform best under a fixed probing budget. This repository contains the data generation, training, evaluation, and analysis code used in the accompanying paper *GeoPAS: Geometric Probing for Algorithm Selection in Continuous Black-Box Optimisation*.

## Repository layout

```text
GeoPAS/
├── functions/
│   ├── model.py
│   └── model_interface.py
├── data_generation/
│   ├── performances/
│   │   ├── ERT_cal.ipynb
│   │   └── relert.csv
│   └── plots/
│       ├── auxiliary_functions.py
│       ├── plot_generation_soo_extensive.py
│       ├── plot_check.ipynb
│       └── AS_BBOB_SOO.code-workspace
├── train_parallel.py
├── train.sh
├── analysis.ipynb
├── concatenate_over_parameters.ipynb
└── robustness_over_budget.ipynb
```
<!-- 
## Main files

- `functions/model.py`: GeoPAS model definition
- `functions/model_interface.py`: dataset loading, training, evaluation, and metrics
- `data_generation/performances/ERT_cal.ipynb`: builds the relERT table
- `data_generation/performances/relert.csv`: relERT labels used by training and analysis
- `data_generation/plots/plot_generation_soo_extensive.py`: generates multi-view `.npz` data
- `train.sh`: sweep wrapper for the current experiment grid
- `train_parallel.py`: training and evaluation entry point
- `concatenate_over_parameters.ipynb`: aggregates result CSVs across runs
- `robustness_over_budget.ipynb`: summarizes results over resolution, number of views, and budget
- `analysis.ipynb`: validation and failure-mode analysis -->

## Setup

```bash
conda env create -f environment.yaml
conda activate as_bbo
```

## Pipeline

### 1. Build the relERT table

`data_generation/performances/ERT_cal.ipynb` produces the table, which is also given as `data_generation/performances/relert.csv`.

### 2. Generate multi-slice `.npz` data

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
python data_generation/plots/plot_generation_soo_extensive.py
```

will write data under:

```text
$PROJECT_ROOT/data/bbob_by_deepela/maxscale_0.7_logscale_false/
```

depending on the setting.

### 3. Train and evaluate

To run the current sweep:

```bash
bash train.sh
```

To point training explicitly to the generated data:

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
DATA_ROOT="$PROJECT_ROOT/data/bbob_by_deepela/maxscale_0.7_logscale_false" \
bash train.sh
```

Outputs are written under:

```text
$PROJECT_ROOT/results/bbob_by_deepela/results/bbob/
```

including a summary table and a dataframe of model outputs. 

### 4. Aggregate results

If results over multiple parameter settings are obtained, use `concatenate_over_parameters.ipynb` to aggregate them into protocol-wise tables`AS_mean_median_p90__{LPO,LIO,RANDOM}__ALL_RUNS.csv`, and then to `AS_mean_median_p90__MERGED__ALL_RUNS.csv`. 

### 5. Analyses

- `robustness_over_budget.ipynb` summarises results over resolution and number of views. 
- `analysis.ipynb`: validation and failure-mode analysis

## Path overrides

The main path overrides used by `train.sh` are:

```bash
PROJECT_ROOT=/path/to/AS_BBO_REBUILT \
RESULTS_ROOT=/path/to/results/bbob_by_deepela/results \
DATA_ROOT=/path/to/generated_npz_root \
OUT_DIR_BASE=/path/to/results/bbob \
TB_LOG_DIR=/path/to/results/tensorboard \
bash train.sh
```

The notebooks use the same default root resolution and also honor `PROJECT_ROOT` and `RESULTS_ROOT`.

## Direct training entry point

To bypass `train.sh`:

```bash
python train_parallel.py \
  --protocol all \
  --csv data_generation/performances/relert.csv \
  --data-root /path/to/data_root \
  --out-dir /path/to/results
```
