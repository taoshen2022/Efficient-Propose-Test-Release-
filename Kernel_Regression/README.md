# Kernel Regression Experiments

This folder contains the cleaned code for the kernel-regression task.

The package has two parts:

1. **Synthetic one-dimensional kernel-regression simulations**.
2. **Real-data California Housing experiment** using kernel regression.

The main methods are:

- `Non-private`: non-private Nadaraya--Watson kernel regression.
- `ePTR`: ePTR Nadaraya--Watson kernel regression.
- `WPE`: a simple Haar-wavelet private estimator baseline.

For the California Housing real-data experiment, the script also saves a public midpoint baseline and a DP mean baseline for diagnostics, but the main plots show `non-private`, `ePTR`, and `WPE`.

## Files

### Core files

- `kernel_data.py`: synthetic data generator for the one-dimensional kernel-regression simulation.
- `eptr_kernel_nw.py`: one-dimensional Gaussian-kernel Nadaraya--Watson estimator, ePTR version, and WPE baseline.

### Synthetic experiments

- `exp_kernel_vary_epsilon.py`: mean pointwise squared error versus privacy budget.
- `exp_kernel_vary_n.py`: mean pointwise squared error versus sample size.
- `exp_kernel_vary_beta_a.py`: local-density stress test with `X ~ Beta(a, a)`.

### Real-data experiment

- `california_kernel_realdata.py`: California Housing kernel-regression experiment.

### Environment

- `requirements.txt`: minimal Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```

The California Housing script uses `sklearn.datasets.fetch_california_housing` by default. If the dataset is not already cached, scikit-learn may need internet access to download it. You can also provide a local CSV file with `--data-path`.

## Synthetic experiments

### Error versus privacy budget

```bash
python exp_kernel_vary_epsilon.py \
  --reps 500 \
  --n 2000 \
  --eps-grid "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0" \
  --outdir ./KernelRegression/results_vs_eps
```

Outputs:

- `raw_vs_eps.csv`
- `summary_vs_eps.csv`
- `plot_vs_eps.png`

### Error versus sample size

```bash
python exp_kernel_vary_n.py \
  --reps 500 \
  --eps 1.5 \
  --n-grid "500,1000,1500,2000,2500,3000,3500,4000,4500,5000" \
  --outdir ./KernelRegression/results_vs_n
```

Outputs:

- `raw_vs_n.csv`
- `summary_vs_n.csv`
- `plot_vs_n.png`

### Local-density stress test

```bash
python exp_kernel_vary_beta_a.py \
  --reps 100 \
  --n 2000 \
  --eps 1.5 \
  --a-grid "0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,2.9,3.2,3.5" \
  --outdir ./KernelRegression/results_beta_vs_a
```

Outputs:

- `raw_beta_vs_a.csv`
- `summary_beta_vs_a.csv`
- `plot_beta_vs_a.png`

## Real-data experiment: California Housing

The real-data script predicts California median house value using kernel regression. By default, the response is `MedHouseVal`, measured in units of 100,000 dollars.

Default fixed features:

```text
medinc, latitude, longitude, houseage
```

These are scaled using fixed public ranges and then mapped to `[0,1]`. The response is clipped and scaled using the public range `[0,5]`, so no train-split mean, standard deviation, minimum, or maximum is used in the default preprocessing.

### Run with scikit-learn fetch

```bash
python california_kernel_realdata.py \
  --reps 50 \
  --train-frac 0.2 \
  --kernel-eval-size 20 \
  --outdir ./KernelRegression/california_kernel_outputs
```

### Run with a local CSV

Use this if `fetch_california_housing` cannot download the data or if you already saved the dataset locally.

```bash
python california_kernel_realdata.py \
  --data-path ./california_housing.csv \
  --target-column MedHouseVal \
  --reps 50 \
  --train-frac 0.2 \
  --kernel-eval-size 20 \
  --outdir ./KernelRegression/california_kernel_outputs
```

The local CSV should contain the feature columns after normalization, for example:

```text
MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
```

Column names are normalized internally, so names like `MedInc`, `medinc`, and `Med Inc` are handled consistently.

Outputs:

- `raw_results.csv`
- `summary_results.csv`
- `mse_vs_eps_standardized_response.png`
- `mse_vs_eps_original_scale.png`
- `release_rate_vs_eps.png`
- `available_features_after_preprocessing.csv`
- `public_feature_ranges.csv`
- `feature_selection_by_rep.csv`
- `selected_features_by_rep.csv`
- `run_metadata.csv`
