This repository contains simulation code for Swarm Learning with Random Forests under feature heterogeneity (partially overlapping feature sets across sites). It reproduces the experiments and plots from the accompanying paper.

## Overview
- **Simulation**: Simulates cross-silo Swarm Learning with multiple peers.
- **Training**: Trains local Random Forests per peer and pools trees into a global forest.
- **Inference**: Evaluates strategies to handle missing splits caused by non-overlapping features.
- **Analysis**: Saves per-iteration scores and generates plots from saved results.


## Datasets

All datasets must be placed in the `data/` directory.

### Synthetic Datasets
The following synthetic datasets are included in this repository:

| Dataset Name | Target Path |
| :--- | :--- |
| **s_500_num** | `apps/data/s_500_num/s_500_num.csv` |
| **s_500_cat** | `apps/data/s_500_cat/s_500_cat.csv` |
| **s_500_mix** | `apps/data/s_500_mix/s_500_mix.csv` |

### Public Datasets
Please download the following datasets and place the CSV files in their respective folders as indicated below.

| Dataset Name | Target Path | Source Link |
| :--- | :--- | :--- |
| **glioma** | `apps/data/glioma/glioma.csv` | https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset |
| **thyroid** | `apps/data/thyroid/Thyroid_Diff.csv` | https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence |
| **diabetes** | `apps/data/diabetes/diabetes.csv` | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download |
| **gallstone** | `apps/data/gallstone/gallstone.csv` | https://archive.ics.uci.edu/dataset/1150/gallstone-1 |
| **cdc_binary5050_stratified** | `apps/data/cdc_binary5050_stratified/diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators |
| **cdc_3class_balanced** | `apps/data/cdc_3class_balanced/diabetes_012_health_indicators_BRFSS2015.csv` | https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators |

*Note: Ensure downloaded filenames match the Target Path column.*

## Installation

### Experiment Environment
Set up the primary environment for running simulations:

```
python -m venv .venv_exp
source .venv_exp/bin/activate
python -m pip install -r requirements.txt
```

### Plotting Environment
To isolate dependencies (e.g., `Orange3`), a separate environment for plotting is recommended:

```
python -m venv .venv_plot
source .venv_plot/bin/activate
python -m pip install -r requirements-plots.txt
```

## Running Experiments

### Run a Single Experiment
Navigate to the `apps` directory and use `run.py` to execute a specific scenario.
```
# Activate environment (if not already active)
source .venv_exp/bin/activate

# Enter the application folder
cd apps
python run.py \
  --dataset cdc_binary5050_stratified \
  --nodes 4 \
  --jaccard 0.3 \
  --method marginal_prediction
```
**Arguments:**
- `--dataset`: Dataset identifier (see Datasets section).
- `--nodes`: Number of participating peers (e.g., 2, 3, 4, 6).
- `--jaccard`: Target feature overlap as Jaccard similarity (0.0 - 1.0).
- `--method`: Heterogeneity handling strategy (see Methods section).

### Run Full Benchmark
Use `run_flat.sh` to reproduce all experiments defined in the paper.

```
chmod +x run_flat.sh
MAX_JOBS=64 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
./run_flat.sh
```

*Note: Adjust `MAX_JOBS` according to available CPU cores.*

### Methods

The following methods are supported for handling missing features during inference:

- `baseline_intersection`: Baseline using only the global intersection of features.
- `feasible_path_voting`: Keeps trees whose sample path uses only available features.
- `model_imputation`: Imputation using local models.
- `probabilistic_routing`: Routing based on training sample distribution.
- `informed_probabilistic_routing`: Probabilistic routing enhanced with imputed values.
- `marginal_prediction`: Prediction based on marginal distributions.
- `surrogate_splits`: Surrogate splitting based on overlapping features.
- `park_et_al`: Park et al. approach.

## Evaluation and Plotting

Experiment results are saved as `.npy` files in the `results/` directory. To generate the plots presented in the paper:
```
# Activate plotting environment
source .venv_plot/bin/activate

# Generate plots
python make_plots.py
```
Generated figures will be saved to the `plots/` directory.
