# GeoPAS

GeoPAS contains the training and analysis pipeline for algorithm selection on BBOB using multi-view two-dimensional slices of black-box functions.

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