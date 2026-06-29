# Linear Regression Experiments

This folder contains the cleaned code for the differentially private linear-regression experiments.
It includes both synthetic simulations and a real-data experiment based on the UCI Wine Quality data.

## Files kept in this folder

### Core synthetic-data code

- `linear_data.py`: generates the Gaussian linear-regression simulation data.
- `eptr_linear_regression.py`: ePTR linear-regression estimator.
- `dp_ols_sheffet_alg1.py`: DPJL baseline based on private Johnson-Lindenstrauss projection.
- `dp_linear_regression_cai2021.py`: DPGD baseline based on private noisy gradient descent.
- `dp_functional_mechanism_lr.py`: Functional Mechanism baseline.
- `linear_regression_experiment_utils.py`: common utilities for running methods, summarizing results, and plotting.

### Synthetic experiment drivers

- `exp_lr_vary_epsilon.py`: synthetic experiment with fixed sample size and varying privacy budget.
- `exp_lr_vary_n.py`: synthetic experiment with fixed privacy budget and varying sample size.

### Real-data experiment driver

- `wine_quality_linear_eps.py`: Wine Quality real-data experiment with varying privacy budget.

The previous TukeyEM script is not included in the cleaned main package because it is not used in the current reported comparisons. The current main comparisons are:

- non-private OLS/ridge;
- ePTR;
- DPJL;
- DPGD;
- Functional Mechanism.

## Installation

Create an environment and install the minimal requirements:

```bash
pip install -r requirements.txt
```

The code only requires `numpy`, `pandas`, and `matplotlib`.

## Synthetic simulation: vary privacy budget

Run:

```bash
python exp_lr_vary_epsilon.py \
  --outdir results_vary_epsilon \
  --reps 50 \
  --n 8000 \
  --p 10 \
  --tau 1 \
  --n_test 100000
```

Main outputs:

- `vary_epsilon_raw.csv`: raw replication-level results.
- `vary_epsilon_mse_summary.csv`: summary of test MSE.
- `vary_epsilon_signal_mse_summary.csv`: summary of signal prediction error.
- `vary_epsilon_param_err_summary.csv`: summary of parameter error.
- `vary_epsilon_mse.png` and `vary_epsilon_mse.pdf`: test-MSE plot.

## Synthetic simulation: vary sample size

Run:

```bash
python exp_lr_vary_n.py \
  --outdir results_vary_n \
  --reps 50 \
  --epsilon 4.0 \
  --p 10 \
  --tau 1 \
  --n_test 100000
```

Main outputs:

- `vary_n_raw.csv`: raw replication-level results.
- `vary_n_mse_summary.csv`: summary of test MSE.
- `vary_n_signal_mse_summary.csv`: summary of signal prediction error.
- `vary_n_param_err_summary.csv`: summary of parameter error.
- `vary_n_mse.png` and `vary_n_mse.pdf`: test-MSE plot.

## Real-data experiment: Wine Quality

The real-data experiment uses the Wine Quality data. Put the two CSV files in this structure:

```text
wine+quality/
  winequality-red.csv
  winequality-white.csv
```

Then run:

```bash
python wine_quality_linear_eps.py \
  --red-path ./wine+quality/winequality-red.csv \
  --white-path ./wine+quality/winequality-white.csv \
  --outdir wine_quality_linear_outputs \
  --reps 500 \
  --train-frac 0.8
```

Alternatively, if you use one combined CSV file, run:

```bash
python wine_quality_linear_eps.py \
  --data-path ./wine_quality_combined.csv \
  --outdir wine_quality_linear_outputs \
  --reps 500 \
  --train-frac 0.8
```

By default, the real-data script uses the following seven linear features:

```text
alcohol, volatile acidity, density, chlorides, free sulfur dioxide, residual sugar, pH
```

The target variable is `quality`. The response is scaled by public constants:

```text
y_scaled = (y - 5) / 5
```

The test MSE is reported on the original wine-quality scale.

The privacy budget is parameterized as `epsilon = 2^t`, and the x-axis in the plot is `log2(epsilon)`.

Main outputs:

- `raw_results.csv`: raw replication-level results.
- `summary_results.csv`: average test MSE and standard errors.
- `linear_eptr_diagnostics_summary.csv`: ePTR release probability and stability diagnostics.
- `run_metadata.csv`: experiment settings.
- `mse_vs_eps.png`: main MSE plot.
- `mse_vs_eps_no_dpgd.png`: MSE plot excluding DPGD.
- `release_rate_vs_eps.png`: ePTR release-rate plot.
