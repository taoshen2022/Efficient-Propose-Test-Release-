# Bayes Classification Experiments

This folder contains Bayes-classification experiments for comparing a non-private Gaussian naive Bayes classifier with several privacy-preserving baselines. It includes both synthetic simulations and one real-data experiment on the UCI credit-default dataset.

## Files

- `bayes_plot_eps.py`: synthetic experiment over a grid of privacy budgets `epsilon`, with fixed training sample size `n`.
- `bayes_plot_n.py`: synthetic experiment over a grid of training sample sizes `n`, with fixed privacy budget `epsilon`.
- `bayes_plot_pimin.py`: synthetic experiment over a grid of minimum class probabilities `pi_min`, with fixed `n` and `epsilon`.
- `credit_default_bayes_eps.py`: real-data experiment on the UCI Default of Credit Card Clients dataset, over a grid of privacy budgets.
- `bayes_data.py`: reusable Gaussian Bayes data-generation utilities.
- `Bayes_helper.py`: reusable implementations of the Bayes estimators and evaluation helpers.
- `requirements.txt`: minimal Python package requirements.
- `data/default of credit card clients.xls`: local copy of the real credit-default dataset used by `credit_default_bayes_eps.py`.

The three synthetic plotting scripts are self-contained, so they can be run directly without importing the helper files. The helper files are retained for modular checks, smaller experiments, or future refactoring.

## Compared methods

The main scripts compare four methods:

- `Nonprivate`: empirical Gaussian naive Bayes without privacy noise.
- `ePTR`: the proposed ePTR-style private Bayes estimator.
- `DPNB`: direct Laplace-noise baseline for private naive Bayes.
- `DPNBSS`: Cauchy-type direct baseline using a simple sensitivity proxy.

Note: `Bayes_helper.py` also contains an LDP Bayes helper, but the LDP baseline is not included in the default plotting scripts.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

The real-data script reads an `.xls` file, so `xlrd` is included in `requirements.txt`. If you instead use an `.xlsx` file, `openpyxl` may also be needed depending on your local pandas installation.

## Synthetic data-generating model

The synthetic scripts simulate a `K=3` Gaussian Bayes classification problem. Labels are sampled from a class-probability vector `mu`, and conditional on the label, features are generated from a Gaussian distribution with identity covariance and class-specific mean vectors. The default dimension is `p=10`, and the default class separation is controlled by `delta_sep=3.0`.

The default metric is balanced test error, which is useful because the default class probabilities are imbalanced. Ordinary misclassification error is also supported through `--metric misclassification`.

## Synthetic experiments

Run error versus privacy budget:

```bash
python revised_bayes_plot_eps.py \
  --reps 500 \
  --n-train 5000 \
  --n-test 100000 \
  --metric balanced_error
```

Run error versus training sample size:

```bash
python revised_bayes_plot_n.py \
  --reps 500 \
  --eps 1.5 \
  --n-test 100000 \
  --metric balanced_error
```

Run error versus minimum class probability:

```bash
python revised_bayes_plot_pimin.py \
  --reps 500 \
  --n-train 5000 \
  --n-test 100000 \
  --eps 1.5 \
  --metric balanced_error
```

For a quick check, use smaller values:

```bash
python revised_bayes_plot_eps.py --reps 2 --n-train 200 --n-test 1000
python revised_bayes_plot_n.py --reps 2 --n-test 1000
python revised_bayes_plot_pimin.py --reps 2 --n-train 200 --n-test 1000
```

Default synthetic outputs are saved under `./Classification/`.

## Real-data experiment: credit-default classification

The real-data script is:

```bash
credit_default_bayes_eps.py
```

It uses the UCI Default of Credit Card Clients dataset. The response is `default.payment.next.month`, and the task is binary classification. The script builds engineered credit-risk features from credit limit, age, repayment status, bill amounts, payment amounts, and optionally categorical demographic variables.

The default local data path is:

```bash
./data/default of credit card clients.xls
```

Run the real-data experiment with default settings:

```bash
python credit_default_bayes_eps.py
```

Equivalent explicit command:

```bash
python credit_default_bayes_eps.py \
  --data-path "./data/default of credit card clients.xls" \
  --reps 200 \
  --train-frac 0.2 \
  --feature-mode selected \
  --standardize train \
  --metric balanced_error
```

For a quick smoke test:

```bash
python credit_default_bayes_eps.py \
  --reps 1 \
  --eps-powers 0 1 \
  --shade none
```

Important options for the real-data script:

- `--data-path`: path to the credit-default `.xls`, `.xlsx`, or `.csv` file.
- `--reps`: number of random train/test repetitions.
- `--train-frac`: training fraction in each repetition.
- `--eps-powers`: values of `t` where `epsilon = 2^t`. The x-axis is `log2(epsilon)`.
- `--feature-mode`: `selected`, `compact`, or `expanded`.
- `--standardize`: `train` or `none`. The default uses training-split standardization.
- `--select-k`: if positive, selects the top `k` features within the training split using standardized mean difference.
- `--metric`: `balanced_error` or `misclassification`.
- `--prediction-prior`: `estimated` or `uniform` class prior in prediction.
- `--shade`: `se`, `sd`, or `none` for error bars.

Real-data outputs are saved by default under:

```bash
./Classification/credit_default_eptr_bayes_eps/
```

The main saved files are:

- `credit_default_bayes_error_vs_eps_balanced_error_raw.csv`
- `credit_default_bayes_error_vs_eps_balanced_error_summary.csv`
- `credit_default_eptr_components_raw.csv`
- `credit_default_eptr_components_summary.csv`
- `credit_default_selected_features_raw.csv`
- `credit_default_selected_features_summary.csv`
- `credit_default_feature_names.csv`
- `credit_default_experiment_settings.csv`
- `credit_default_bayes_error_vs_eps_balanced_error.png`

## Main options for synthetic scripts

Common options:

- `--reps`: number of Monte Carlo replications.
- `--n-train`: training sample size, where applicable.
- `--n-test`: test sample size.
- `--eps`: privacy budget for the fixed-epsilon experiments.
- `--delta`: approximate-DP failure probability used by ePTR.
- `--delta-sep`: class-separation strength in the Gaussian model.
- `--p`: feature dimension.
- `--Rx`: clipping radius for feature vectors.
- `--c0`: lower-bound parameter used by ePTR.
- `--eps-split-counts`: fraction of the direct-baseline privacy budget assigned to class counts.
- `--metric`: either `balanced_error` or `misclassification`.
- `--outdir`: output directory.

## Outputs

Each script saves:

- a raw result CSV containing one row per method, parameter value, and replication;
- a summary CSV containing means, standard deviations, and quantiles or standard errors;
- a PNG figure.

Examples for synthetic scripts:

- `bayes_error_vs_eps_balanced_error_raw.csv`
- `bayes_error_vs_eps_balanced_error_summary.csv`
- `bayes_error_vs_eps_balanced_error.png`
