from __future__ import annotations

import argparse
import io
import time
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Dataset loading and feature engineering
# ============================================================

UCI_DATASET_ID = 350
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

X_TO_NAME = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": "default.payment.next.month",
}

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def _clean_colname(c: object) -> str:
    return str(c).strip().replace(" ", ".")


def normalize_credit_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_colname(c) for c in df.columns]

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.rename(columns={c: X_TO_NAME[c] for c in df.columns if c in X_TO_NAME})

    for c in list(df.columns):
        lowered = c.lower()
        if lowered in {"default.payment.next.month", "default.payment.next.month."}:
            df = df.rename(columns={c: "default.payment.next.month"})
        elif lowered.replace(".", " ") == "default payment next month":
            df = df.rename(columns={c: "default.payment.next.month"})

    for c in df.columns:
        if c != "ID":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_credit_default(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the UCI Default of Credit Card Clients dataset.

    Priority:
      1. local --data-path if provided;
      2. ucimlrepo fetch_ucirepo(id=350), if installed;
      3. direct download of the UCI zip archive.
    """
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find data_path: {path}")

        if path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path, header=1)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError("--data-path must be a .xls, .xlsx, .csv, or .txt file")

        return normalize_credit_columns(df)

    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore

        ds = fetch_ucirepo(id=UCI_DATASET_ID)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        return normalize_credit_columns(df)
    except Exception as e_ucimlrepo:
        print("ucimlrepo loading failed; trying direct UCI zip download.")
        print(f"ucimlrepo error: {e_ucimlrepo}")

    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        excel_files = [
            name for name in zf.namelist()
            if name.lower().endswith((".xls", ".xlsx"))
        ]
        if not excel_files:
            raise RuntimeError("No Excel file found inside the UCI zip archive.")

        with zf.open(excel_files[0]) as f:
            df = pd.read_excel(f, header=1)

    return normalize_credit_columns(df)


def signed_log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def build_credit_features(
    df: pd.DataFrame,
    feature_mode: str = "compact",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = normalize_credit_columns(df)
    label_col = "default.payment.next.month"

    required = (
        ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
        + PAY_STATUS_COLS
        + BILL_COLS
        + PAY_AMT_COLS
        + [label_col]
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

    df = df[required].dropna().copy()

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).clip(1, 4)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).clip(1, 3)
    df["SEX"] = df["SEX"].clip(1, 2)

    y = df[label_col].astype(int).to_numpy()

    money_scale = np.log1p(1_000_000.0)
    payment_total_scale = np.log1p(5_000_000.0)

    feat = pd.DataFrame(index=df.index)

    feat["limit_bal_log"] = np.log1p(df["LIMIT_BAL"].clip(lower=0)) / money_scale
    feat["age_scaled"] = (df["AGE"].clip(18, 100) - 40.0) / 20.0

    pay_status = df[PAY_STATUS_COLS].clip(-2, 8)
    bills = df[BILL_COLS]
    pay_amts = df[PAY_AMT_COLS].clip(lower=0)

    feat["repay_status_recent"] = pay_status["PAY_0"] / 8.0
    feat["repay_status_avg"] = pay_status.mean(axis=1) / 8.0
    feat["repay_status_max"] = pay_status.max(axis=1) / 8.0
    feat["frac_months_delay"] = (pay_status > 0).mean(axis=1)

    feat["bill_recent_log"] = signed_log1p(bills["BILL_AMT1"]) / money_scale
    feat["bill_avg_log"] = signed_log1p(bills.mean(axis=1)) / money_scale
    feat["bill_trend_log"] = signed_log1p(bills["BILL_AMT1"] - bills["BILL_AMT6"]) / money_scale
    feat["utilization_avg"] = (
        bills.mean(axis=1) / np.maximum(df["LIMIT_BAL"], 1.0)
    ).clip(-2, 2) / 2.0

    feat["payment_amt_recent_log"] = np.log1p(pay_amts["PAY_AMT1"]) / money_scale
    feat["payment_amt_avg_log"] = np.log1p(pay_amts.mean(axis=1)) / money_scale
    feat["payment_amt_total_log"] = np.log1p(pay_amts.sum(axis=1)) / payment_total_scale
    feat["payment_to_bill_ratio"] = (
        pay_amts.sum(axis=1) / (np.abs(bills).sum(axis=1) + 1.0)
    ).clip(0, 5) / 5.0

    selected_core = [
        "limit_bal_log",
        "age_scaled",
        "repay_status_recent",
        "repay_status_avg",
        "repay_status_max",
        "frac_months_delay",
        "utilization_avg",
        "payment_amt_recent_log",
        "payment_amt_avg_log",
        "payment_amt_total_log",
        "payment_to_bill_ratio",
    ]

    if feature_mode == "selected":
        feat = feat[selected_core].copy()

    elif feature_mode == "expanded":
        for c in PAY_STATUS_COLS:
            feat[f"{c.lower()}_scaled"] = pay_status[c] / 8.0

        for c in BILL_COLS:
            feat[f"{c.lower()}_slog"] = signed_log1p(bills[c]) / money_scale

        for c in PAY_AMT_COLS:
            feat[f"{c.lower()}_log"] = np.log1p(pay_amts[c]) / money_scale

        dummies = pd.get_dummies(
            df[CAT_COLS].astype(int).astype(str),
            prefix=CAT_COLS,
            dtype=float,
        )
        feat = pd.concat([feat, dummies], axis=1)

    elif feature_mode == "compact":
        dummies = pd.get_dummies(
            df[CAT_COLS].astype(int).astype(str),
            prefix=CAT_COLS,
            dtype=float,
        )
        feat = pd.concat([feat, dummies], axis=1)

    else:
        raise ValueError("feature_mode must be 'selected', 'compact', or 'expanded'.")

    X = feat.astype(float).to_numpy()
    feature_names = list(feat.columns)

    return X, y, feature_names


# ============================================================
# Bayes/ePTR helpers
# ============================================================

def clip_to_ball(X: np.ndarray, Rx: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if Rx is None or np.isinf(Rx):
        return X.copy()

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.minimum(1.0, Rx / np.maximum(norms, 1e-12))

    return X * scale


def fit_empirical_bayes(
    X: np.ndarray,
    y: np.ndarray,
    K: Optional[int] = None,
):
    if K is None:
        K = int(np.max(y)) + 1

    n, p = X.shape
    counts = np.bincount(y, minlength=K)
    mu_hat = counts / n

    means_hat = np.zeros((K, p), dtype=float)
    for k in range(K):
        if counts[k] > 0:
            means_hat[k] = X[y == k].mean(axis=0)

    return mu_hat, means_hat, counts


def sigmoid(z: float) -> float:
    z = np.clip(z, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def sample_cauchy(
    rng: np.random.Generator,
    loc: float = 0.0,
    scale: float = 1.0,
    size=None,
):
    return loc + scale * rng.standard_cauchy(size=size)


def random_fallback_parameter(
    K: int,
    p: int,
    c0: float,
    Rx: float,
    rng: np.random.Generator,
):
    mu_rand = rng.dirichlet(np.ones(K))
    mu_rand = np.maximum(mu_rand, c0)
    mu_rand = mu_rand / mu_rand.sum()

    means_rand = rng.normal(loc=0.0, scale=Rx, size=(K, p))
    means_rand = clip_to_ball(means_rand, Rx)

    return np.concatenate([mu_rand, means_rand.ravel()])


def etpr_release_vector(
    theta_hat: np.ndarray,
    gamma: float,
    alpha: float,
    eps: float,
    delta: float,
    K: int,
    p: int,
    c0: float,
    Rx: float,
    rng: np.random.Generator,
):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1).")

    M = 1.0 + (2.0 / eps) * np.log(max(1.0 / delta, 1.0 / eps))
    p_release = sigmoid(0.5 * eps * (gamma - M))
    released = bool(rng.uniform() < p_release)

    noise_sd = (2.0 * alpha / eps) * np.sqrt(2.0 * np.log(1.25 / delta))

    if released:
        theta_tilde = theta_hat + rng.normal(
            loc=0.0,
            scale=noise_sd,
            size=theta_hat.shape,
        )
    else:
        theta_tilde = random_fallback_parameter(
            K=K,
            p=p,
            c0=c0,
            Rx=Rx,
            rng=rng,
        )

    info = {
        "released": released,
        "M": float(M),
        "p_release": float(p_release),
        "noise_sd": float(noise_sd),
    }

    return theta_tilde, info


def fit_bayes_etpr(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    delta: float,
    Rx: float,
    c0: float,
    K: int = 2,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)

    n, p = X.shape
    X_clip = clip_to_ball(X, Rx)

    mu_hat, means_hat, counts = fit_empirical_bayes(X_clip, y, K=K)

    gamma = max(float(counts.min() - c0 * n - 1.0), 0.0)
    alpha = (2.0 / n) * np.sqrt(2.0 * Rx**2 / c0**2 + 2.0)

    theta_hat = np.concatenate([mu_hat, means_hat.ravel()])

    theta_tilde, info = etpr_release_vector(
        theta_hat=theta_hat,
        gamma=gamma,
        alpha=alpha,
        eps=eps,
        delta=delta,
        K=K,
        p=p,
        c0=c0,
        Rx=Rx,
        rng=rng,
    )

    mu_tilde = theta_tilde[:K].copy()
    means_tilde = theta_tilde[K:].reshape(K, p).copy()

    mu_tilde = np.maximum(mu_tilde, c0)
    mu_tilde = mu_tilde / mu_tilde.sum()
    means_tilde = clip_to_ball(means_tilde, Rx)

    return {
        "mu_tilde": mu_tilde,
        "means_tilde": means_tilde,
        "mu_hat": mu_hat,
        "means_hat": means_hat,
        "counts_hat": counts,
        "gamma": float(gamma),
        "alpha": float(alpha),
        "M": info["M"],
        "p_release": info["p_release"],
        "released": info["released"],
        "noise_sd": info["noise_sd"],
    }


def fit_bayes_direct_wi13(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: int = 2,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
):
    rng = np.random.default_rng(seed)

    n, p = X.shape
    X_clip = clip_to_ball(X, Rx)

    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_lap_scale = 2.0 / eps_count

    counts_tilde = counts_hat.astype(float) + rng.laplace(
        0.0,
        count_lap_scale,
        size=K,
    )
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)

    mean_lap_scales = np.zeros(K)
    means_tilde = means_hat.copy()

    for k in range(K):
        sens_mean_k = 2.0 * Rx / (counts_hat[k] + 1.0)
        scale_mean_k = sens_mean_k / eps_per_mean_coord
        mean_lap_scales[k] = scale_mean_k
        means_tilde[k, :] = means_hat[k, :] + rng.laplace(
            0.0,
            scale_mean_k,
            size=p,
        )

    means_tilde = clip_to_ball(means_tilde, Rx)

    return {
        "mu_tilde": mu_tilde,
        "means_tilde": means_tilde,
        "count_lap_scale": float(count_lap_scale),
        "mean_lap_scale_min": float(mean_lap_scales.min()),
        "mean_lap_scale_max": float(mean_lap_scales.max()),
    }


def fit_bayes_direct_cauchy(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: int = 2,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
):
    rng = np.random.default_rng(seed)

    n, p = X.shape
    X_clip = clip_to_ball(X, Rx)

    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_cauchy_scale = np.sqrt(2.0) * 2.0 / eps_count

    counts_tilde = counts_hat.astype(float) + sample_cauchy(
        rng,
        0.0,
        count_cauchy_scale,
        size=K,
    )
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)

    mean_cauchy_scales = np.zeros(K)
    means_tilde = means_hat.copy()

    for k in range(K):
        sens_mean_k = 2.0 * Rx / (counts_hat[k] + 1.0)
        scale_mean_k = np.sqrt(2.0) * sens_mean_k / eps_per_mean_coord
        mean_cauchy_scales[k] = scale_mean_k
        means_tilde[k, :] = means_hat[k, :] + sample_cauchy(
            rng,
            0.0,
            scale_mean_k,
            size=p,
        )

    means_tilde = clip_to_ball(means_tilde, Rx)

    return {
        "mu_tilde": mu_tilde,
        "means_tilde": means_tilde,
        "count_cauchy_scale": float(count_cauchy_scale),
        "mean_cauchy_scale_min": float(mean_cauchy_scales.min()),
        "mean_cauchy_scale_max": float(mean_cauchy_scales.max()),
    }


def predict_naive_bayes(
    X: np.ndarray,
    mu: np.ndarray,
    means: np.ndarray,
    prediction_prior: str = "estimated",
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    means = np.asarray(means, dtype=float)

    K = len(mu)

    if prediction_prior == "uniform":
        mu_pred = np.ones(K) / K
    elif prediction_prior == "estimated":
        mu_pred = mu.copy()
    else:
        raise ValueError("prediction_prior must be either 'estimated' or 'uniform'.")

    mu_pred = np.maximum(mu_pred, 1e-12)
    mu_pred = mu_pred / mu_pred.sum()

    scores = X @ means.T - 0.5 * np.sum(means**2, axis=1) + np.log(mu_pred)

    return np.argmax(scores, axis=1)


def misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def balanced_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int = 2,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    errs = []
    for k in range(K):
        idx = y_true == k
        if idx.any():
            errs.append(np.mean(y_pred[idx] != y_true[idx]))

    return float(np.mean(errs))


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    K: int = 2,
) -> float:
    if metric == "misclassification":
        return misclassification_rate(y_true, y_pred)

    if metric == "balanced_error":
        return balanced_error_rate(y_true, y_pred, K=K)

    raise ValueError(f"Unknown metric: {metric}")


# ============================================================
# Experiment, summaries, and plotting
# ============================================================

def stratified_split_indices(
    y: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)

    train_parts = []
    test_parts = []

    for k in np.unique(y):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)

        n_train_k = int(np.floor(train_frac * len(idx)))

        train_parts.append(idx[:n_train_k])
        test_parts.append(idx[n_train_k:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def standardize_fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    standardize: str = "none",
    min_scale: float = 1e-8,
):
    """
    Standardize variables using training-split statistics only.

    standardize="none": no additional scaling.
    standardize="train": subtract training mean and divide by training sd.

    Note: this is useful for diagnosing model fit. For a fully rigorous DP
    experiment, the scaler parameters should be public, fixed in advance, or
    privatized, because they are computed from the private training sample.
    """
    if standardize == "none":
        info = {
            "standardized": 0,
            "mean_abs_center": 0.0,
            "min_scale": 1.0,
            "max_scale": 1.0,
        }
        return X_train.copy(), X_test.copy(), info

    if standardize != "train":
        raise ValueError("standardize must be either 'none' or 'train'.")

    center = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=0)
    scale = np.where(scale < min_scale, 1.0, scale)

    X_train_std = (X_train - center) / scale
    X_test_std = (X_test - center) / scale

    info = {
        "standardized": 1,
        "mean_abs_center": float(np.mean(np.abs(center))),
        "min_scale": float(np.min(scale)),
        "max_scale": float(np.max(scale)),
    }

    return X_train_std, X_test_std, info


def supervised_select_features_train_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    select_k: int = 0,
    selection_method: str = "smd",
):
    """
    Select features using the training split only, then apply the same columns
    to train and test. The selected columns are fixed for all eps values within
    the repetition.

    selection_method="smd" uses absolute standardized mean difference between
    Y=1 and Y=0, which is appropriate for the Gaussian mean-based Bayes rule.

    Note: supervised feature selection uses private labels/features. For a
    fully rigorous DP claim, use feature-mode='selected' as a public/domain
    feature set, or privatize the feature-selection step.
    """
    p = X_train.shape[1]

    if select_k is None or select_k <= 0 or select_k >= p:
        selected_idx = np.arange(p)
        scores = np.full(p, np.nan)
        selected_names = list(feature_names)
        return X_train.copy(), X_test.copy(), selected_idx, scores, selected_names

    if selection_method != "smd":
        raise ValueError("Currently only selection_method='smd' is implemented.")

    y_train = np.asarray(y_train, dtype=int)

    idx0 = y_train == 0
    idx1 = y_train == 1

    if not idx0.any() or not idx1.any():
        selected_idx = np.arange(min(select_k, p))
        scores = np.zeros(p)
    else:
        m0 = X_train[idx0].mean(axis=0)
        m1 = X_train[idx1].mean(axis=0)
        v0 = X_train[idx0].var(axis=0, ddof=0)
        v1 = X_train[idx1].var(axis=0, ddof=0)

        denom = np.sqrt(0.5 * (v0 + v1)) + 1e-12
        scores = np.abs(m1 - m0) / denom

        selected_idx = np.lexsort((np.arange(p), -scores))[:select_k]
        selected_idx = np.sort(selected_idx)

    selected_names = [feature_names[j] for j in selected_idx]

    return X_train[:, selected_idx], X_test[:, selected_idx], selected_idx, scores, selected_names


def binary_diagnostics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    pos = y_true == 1
    neg = y_true == 0

    tpr = np.mean(y_pred[pos] == 1) if np.any(pos) else np.nan
    fpr = np.mean(y_pred[neg] == 1) if np.any(neg) else np.nan

    return {
        "pred_default_rate": float(np.mean(y_pred == 1)),
        "tpr_default": float(tpr),
        "fpr_default": float(fpr),
    }


def evaluate_one_rep_all_eps(
    X: np.ndarray,
    y: np.ndarray,
    eps_grid: Sequence[float],
    rep: int,
    train_frac: float,
    delta: float,
    Rx: float,
    c0: float,
    K: int,
    eps_split_counts: float,
    metric: str,
    prediction_prior: str,
    seed_base: int,
    feature_names: Sequence[str],
    standardize: str,
    select_k: int,
    selection_method: str,
):
    split_seed = seed_base + 10_000_000 + rep

    train_idx, test_idx = stratified_split_indices(
        y,
        train_frac=train_frac,
        seed=split_seed,
    )

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    X_train_sel, X_test_sel, selected_idx, selection_scores, selected_names = (
        supervised_select_features_train_only(
            X_train_raw,
            y_train,
            X_test_raw,
            feature_names=feature_names,
            select_k=select_k,
            selection_method=selection_method,
        )
    )

    X_train, X_test, standardize_info = standardize_fit_transform_train_test(
        X_train_sel,
        X_test_sel,
        standardize=standardize,
    )

    X_test_clip = clip_to_ball(X_test, Rx)

    p = X_train.shape[1]

    error_rows = []
    diag_rows = []
    selected_feature_rows = []

    t0 = time.perf_counter()

    X_train_clip = clip_to_ball(X_train, Rx)
    mu_np, means_np, counts_np = fit_empirical_bayes(X_train_clip, y_train, K=K)

    pred_np = predict_naive_bayes(
        X_test_clip,
        mu_np,
        means_np,
        prediction_prior=prediction_prior,
    )

    err_np_metric = compute_metric(y_test, pred_np, metric=metric, K=K)
    err_np_mis = misclassification_rate(y_test, pred_np)
    err_np_bal = balanced_error_rate(y_test, pred_np, K=K)
    diag_np = binary_diagnostics(y_test, pred_np)

    runtime_nonprivate = time.perf_counter() - t0
    print(
        f"[time] rep={rep + 1}, method=Nonprivate, "
        f"runtime={runtime_nonprivate:.4f}s",
        flush=True,
    )

    for eps_j, eps in enumerate(eps_grid):
        t_log2_eps = float(np.log2(eps))

        error_rows.append({
            "eps": eps,
            "t_log2_eps": t_log2_eps,
            "rep": rep,
            "method": "Nonprivate",
            "error": err_np_metric,
            "misclassification": err_np_mis,
            "balanced_error": err_np_bal,
            "pred_default_rate": diag_np["pred_default_rate"],
            "tpr_default": diag_np["tpr_default"],
            "fpr_default": diag_np["fpr_default"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "p": p,
            "p_original": X.shape[1],
            "select_k": select_k,
            "standardize": standardize,
        })

        t0 = time.perf_counter()
        etpr = fit_bayes_etpr(
            X=X_train,
            y=y_train,
            eps=eps,
            delta=delta,
            Rx=Rx,
            c0=c0,
            K=K,
            seed=seed_base + rep * 100_000 + eps_j * 101 + 1,
        )

        pred_etpr = predict_naive_bayes(
            X_test_clip,
            etpr["mu_tilde"],
            etpr["means_tilde"],
            prediction_prior=prediction_prior,
        )

        diag_etpr = binary_diagnostics(y_test, pred_etpr)
        runtime_etpr = time.perf_counter() - t0
        print(
            f"[time] rep={rep + 1}, eps={eps:g}, method=ePTR, "
            f"runtime={runtime_etpr:.4f}s",
            flush=True,
        )

        error_rows.append({
            "eps": eps,
            "t_log2_eps": t_log2_eps,
            "rep": rep,
            "method": "ePTR",
            "error": compute_metric(y_test, pred_etpr, metric=metric, K=K),
            "misclassification": misclassification_rate(y_test, pred_etpr),
            "balanced_error": balanced_error_rate(y_test, pred_etpr, K=K),
            "pred_default_rate": diag_etpr["pred_default_rate"],
            "tpr_default": diag_etpr["tpr_default"],
            "fpr_default": diag_etpr["fpr_default"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "p": p,
            "p_original": X.shape[1],
            "select_k": select_k,
            "standardize": standardize,
        })

        counts = etpr["counts_hat"]

        diag_rows.append({
            "eps": eps,
            "t_log2_eps": t_log2_eps,
            "rep": rep,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "p": p,
            "p_original": X.shape[1],
            "select_k": select_k,
            "standardize": standardize,
            "standardized": standardize_info["standardized"],
            "std_mean_abs_center": standardize_info["mean_abs_center"],
            "std_min_scale": standardize_info["min_scale"],
            "std_max_scale": standardize_info["max_scale"],
            "selected_features": ";".join(selected_names),
            "Rx": Rx,
            "c0": c0,
            "delta": delta,
            "count_0": int(counts[0]),
            "count_1": int(counts[1]) if K > 1 else np.nan,
            "min_count": int(np.min(counts)),
            "minority_prop": float(np.min(counts) / len(train_idx)),
            "gamma": etpr["gamma"],
            "alpha": etpr["alpha"],
            "M": etpr["M"],
            "p_release": etpr["p_release"],
            "released": int(etpr["released"]),
            "noise_sd": etpr["noise_sd"],
            "mu_hat_0": float(etpr["mu_hat"][0]),
            "mu_hat_1": float(etpr["mu_hat"][1]) if K > 1 else np.nan,
            "mu_tilde_0": float(etpr["mu_tilde"][0]),
            "mu_tilde_1": float(etpr["mu_tilde"][1]) if K > 1 else np.nan,
        })

        if eps_j == 0:
            for rank, j in enumerate(selected_idx):
                score_j = selection_scores[int(j)]
                selected_feature_rows.append({
                    "rep": rep,
                    "rank_by_original_index": rank,
                    "feature_index": int(j),
                    "feature_name": feature_names[int(j)],
                    "selection_score": float(score_j) if np.isfinite(score_j) else np.nan,
                    "selected": 1,
                    "p_selected": p,
                    "p_original": X.shape[1],
                    "select_k": select_k,
                    "standardize": standardize,
                })
        t0 = time.perf_counter()
        wi13 = fit_bayes_direct_wi13(
            X=X_train,
            y=y_train,
            eps=eps,
            Rx=Rx,
            K=K,
            seed=seed_base + rep * 100_000 + eps_j * 101 + 2,
            eps_split_counts=eps_split_counts,
        )

        pred_wi13 = predict_naive_bayes(
            X_test_clip,
            wi13["mu_tilde"],
            wi13["means_tilde"],
            prediction_prior=prediction_prior,
        )

        diag_wi13 = binary_diagnostics(y_test, pred_wi13)
        runtime_dpnb = time.perf_counter() - t0
        print(
            f"[time] rep={rep + 1}, eps={eps:g}, method=dpnb, "
            f"runtime={runtime_dpnb:.4f}s",
            flush=True,
        )

        error_rows.append({
            "eps": eps,
            "t_log2_eps": t_log2_eps,
            "rep": rep,
            "method": "dpnb",
            "error": compute_metric(y_test, pred_wi13, metric=metric, K=K),
            "misclassification": misclassification_rate(y_test, pred_wi13),
            "balanced_error": balanced_error_rate(y_test, pred_wi13, K=K),
            "pred_default_rate": diag_wi13["pred_default_rate"],
            "tpr_default": diag_wi13["tpr_default"],
            "fpr_default": diag_wi13["fpr_default"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "p": p,
            "p_original": X.shape[1],
            "select_k": select_k,
            "standardize": standardize,
        })

        cauchy = fit_bayes_direct_cauchy(
            X=X_train,
            y=y_train,
            eps=eps,
            Rx=Rx,
            K=K,
            seed=seed_base + rep * 100_000 + eps_j * 101 + 3,
            eps_split_counts=eps_split_counts,
        )

        pred_cauchy = predict_naive_bayes(
            X_test_clip,
            cauchy["mu_tilde"],
            cauchy["means_tilde"],
            prediction_prior=prediction_prior,
        )

        diag_cauchy = binary_diagnostics(y_test, pred_cauchy)

        error_rows.append({
            "eps": eps,
            "t_log2_eps": t_log2_eps,
            "rep": rep,
            "method": "dpnbss",
            "error": compute_metric(y_test, pred_cauchy, metric=metric, K=K),
            "misclassification": misclassification_rate(y_test, pred_cauchy),
            "balanced_error": balanced_error_rate(y_test, pred_cauchy, K=K),
            "pred_default_rate": diag_cauchy["pred_default_rate"],
            "tpr_default": diag_cauchy["tpr_default"],
            "fpr_default": diag_cauchy["fpr_default"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "p": p,
            "p_original": X.shape[1],
            "select_k": select_k,
            "standardize": standardize,
        })

    return error_rows, diag_rows, selected_feature_rows


def summarize_error(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["eps", "method"], as_index=False)

    out = g.agg(
        mean_error=("error", "mean"),
        sd_error=("error", "std"),
        n_rep=("error", "count"),
        mean_misclassification=("misclassification", "mean"),
        mean_balanced_error=("balanced_error", "mean"),
        mean_pred_default_rate=("pred_default_rate", "mean"),
        mean_tpr_default=("tpr_default", "mean"),
        mean_fpr_default=("fpr_default", "mean"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        p=("p", "first"),
        p_original=("p_original", "first"),
        select_k=("select_k", "first"),
    )

    out["sd_error"] = out["sd_error"].fillna(0.0)
    out["se_error"] = out["sd_error"] / np.sqrt(out["n_rep"].clip(lower=1))

    return out.sort_values(["eps", "method"]).reset_index(drop=True)


def summarize_eptr_diag(diag: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "min_count",
        "minority_prop",
        "gamma",
        "alpha",
        "M",
        "p_release",
        "released",
        "noise_sd",
        "mu_hat_1",
        "mu_tilde_1",
    ]

    rows = []

    for eps, sub in diag.groupby("eps"):
        row = {
            "eps": eps,
            "t_log2_eps": float(np.log2(eps)),
            "n_rep": len(sub),
        }

        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[f"mean_{m}"] = float(vals.mean())
            row[f"sd_{m}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"se_{m}"] = row[f"sd_{m}"] / np.sqrt(max(len(vals), 1))

        row["mean_release_rate"] = row["mean_released"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("eps").reset_index(drop=True)


# ============================================================
# Plot style
# ============================================================

METHOD_ORDER = ["Nonprivate", "ePTR", "dpnb", "dpnbss"]

METHOD_LABEL = {
    "Nonprivate": "non-private",
    "ePTR": "ePTR",
    "dpnb": "DPNB",
    "dpnbss": "DPNBSS",
}

METHOD_STYLE = {
    "Nonprivate": {
        "color": "#66A61E",
        "linestyle": ":",
        "marker": "s",
        "open": False,
    },
    "ePTR": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "open": True,
    },
    "dpnb": {
        "color": "#006400",
        "linestyle": "--",
        "marker": "D",
        "open": True,
    },
    "dpnbss": {
        "color": "#56B4E9",
        "linestyle": "-.",
        "marker": "^",
        "open": True,
    },
}


def parse_eps_powers(text_values: Optional[list[float]]) -> list[float]:
    if text_values is not None and len(text_values) > 0:
        return list(text_values)

    return [
        0.0, 0.25, 0.5, 0.75, 1.0,
        1.25, 1.5, 1.75, 2.0, 2.25,
        2.5, 2.75, 3.0, 3.25,
    ]


def _set_log2_eps_axis_ticks(ax, eps_values: np.ndarray) -> None:
    eps_values = np.asarray(eps_values, dtype=float)
    eps_values = eps_values[np.isfinite(eps_values) & (eps_values > 0)]

    if len(eps_values) == 0:
        return

    xvals = np.log2(np.sort(np.unique(eps_values)))
    ax.set_xticks(xvals)
    ax.set_xticklabels([f"{x:g}" for x in xvals])


def plot_error_summary(
    summary: pd.DataFrame,
    out_png: Path,
    metric: str,
    shade: str,
    title: str,
):
    ylabel = (
        "Balanced test error"
        if metric == "balanced_error"
        else "Test misclassification rate"
    )

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("eps")

        if sub.empty:
            continue

        eps_arr = sub["eps"].to_numpy(dtype=float)
        x = np.log2(eps_arr)
        y = sub["mean_error"].to_numpy(dtype=float)

        if shade == "se":
            yerr = sub["se_error"].to_numpy(dtype=float)
        elif shade == "sd":
            yerr = sub["sd_error"].to_numpy(dtype=float)
        else:
            yerr = None

        style = METHOD_STYLE.get(method, {})

        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")
        markerfacecolor = "white" if style.get("open", False) else color

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=METHOD_LABEL.get(method, method),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=1.8,
            linewidth=2.4,
            markersize=7,
            capsize=4,
            capthick=1.6,
            elinewidth=1.6,
        )

    ax.set_xlabel(r"$\log_2(\varepsilon)$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)

    ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        fontsize=10.5,
        loc="best",
    )

    _set_log2_eps_axis_ticks(ax, summary["eps"].dropna().unique().astype(float))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Real-data ePTR Bayes experiment on UCI Default of Credit Card Clients."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/default of credit card clients.xls",
        help="Optional local .xls/.xlsx/.csv path. If omitted, the script tries ucimlrepo or UCI download.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./Classification/credit_default_eptr_bayes_eps",
    )

    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--train-frac", type=float, default=0.2)

    parser.add_argument(
        "--eps-powers",
        type=float,
        nargs="*",
        default=None,
        help="Use epsilon = 2^t. Default t-grid is 0, 0.25, ..., 3.25.",
    )

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--Rx", type=float, default=4.0)
    parser.add_argument("--c0", type=float, default=0.08)
    parser.add_argument("--eps-split-counts", type=float, default=0.5)

    parser.add_argument(
        "--feature-mode",
        type=str,
        default="selected",
        choices=["selected", "compact", "expanded"],
        help="Initial engineered feature pool. Use expanded together with --select-k for train-only filtering.",
    )

    parser.add_argument(
        "--standardize",
        type=str,
        default="train",
        choices=["none", "train"],
        help="none: use fixed engineered scales only; train: standardize by training-split mean/sd.",
    )

    parser.add_argument(
        "--select-k",
        type=int,
        default=0,
        help="If >0, select the top-k features on the training split by absolute standardized mean difference.",
    )

    parser.add_argument("--selection-method", type=str, default="smd", choices=["smd"])

    parser.add_argument(
        "--metric",
        type=str,
        default="balanced_error",
        choices=["balanced_error", "misclassification"],
    )

    parser.add_argument(
        "--prediction-prior",
        type=str,
        default="estimated",
        choices=["estimated", "uniform"],
        help="Use estimated class prior or uniform prior in the Gaussian NB score.",
    )

    parser.add_argument(
        "--shade",
        type=str,
        default="se",
        choices=["se", "sd", "none"],
        help="Use SE/SD/none as error bars. No shaded bands are used.",
    )

    parser.add_argument("--seed-base", type=int, default=20260430)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eps_powers = parse_eps_powers(args.eps_powers)
    eps_grid = [2.0 ** t for t in eps_powers]

    df = load_credit_default(args.data_path)
    X, y, feature_names = build_credit_features(df, feature_mode=args.feature_mode)

    K = int(np.max(y)) + 1
    if K != 2:
        raise ValueError(f"Expected binary response, but got K={K}")

    feature_info = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature_name": feature_names,
    })
    feature_info.to_csv(outdir / "credit_default_feature_names.csv", index=False)

    error_rows = []
    diag_rows = []
    selected_feature_rows = []

    for rep in range(args.reps):
        rep_error, rep_diag, rep_selected = evaluate_one_rep_all_eps(
            X=X,
            y=y,
            eps_grid=eps_grid,
            rep=rep,
            train_frac=args.train_frac,
            delta=args.delta,
            Rx=args.Rx,
            c0=args.c0,
            K=K,
            eps_split_counts=args.eps_split_counts,
            metric=args.metric,
            prediction_prior=args.prediction_prior,
            seed_base=args.seed_base,
            feature_names=feature_names,
            standardize=args.standardize,
            select_k=args.select_k,
            selection_method=args.selection_method,
        )

        error_rows.extend(rep_error)
        diag_rows.extend(rep_diag)
        selected_feature_rows.extend(rep_selected)

        print(f"done rep={rep + 1}/{args.reps}")

    raw = pd.DataFrame(error_rows)
    diag = pd.DataFrame(diag_rows)

    summary = summarize_error(raw)
    diag_summary = summarize_eptr_diag(diag)

    selected_features_raw = pd.DataFrame(selected_feature_rows)

    if not selected_features_raw.empty:
        selected_features_summary = (
            selected_features_raw.groupby(["feature_index", "feature_name"], as_index=False)
            .agg(
                selected_count=("selected", "sum"),
                mean_selection_score=("selection_score", "mean"),
                sd_selection_score=("selection_score", "std"),
            )
            .sort_values(
                ["selected_count", "mean_selection_score"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

        selected_features_summary["selection_rate"] = (
            selected_features_summary["selected_count"] / max(args.reps, 1)
        )

    else:
        selected_features_summary = pd.DataFrame()

    metric_tag = args.metric

    raw_csv = outdir / f"credit_default_bayes_error_vs_eps_{metric_tag}_raw.csv"
    summary_csv = outdir / f"credit_default_bayes_error_vs_eps_{metric_tag}_summary.csv"
    diag_csv = outdir / "credit_default_eptr_components_raw.csv"
    diag_summary_csv = outdir / "credit_default_eptr_components_summary.csv"
    selected_features_raw_csv = outdir / "credit_default_selected_features_raw.csv"
    selected_features_summary_csv = outdir / "credit_default_selected_features_summary.csv"
    plot_png = outdir / f"credit_default_bayes_error_vs_eps_{metric_tag}.png"

    raw.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    diag.to_csv(diag_csv, index=False)
    diag_summary.to_csv(diag_summary_csv, index=False)
    selected_features_raw.to_csv(selected_features_raw_csv, index=False)
    selected_features_summary.to_csv(selected_features_summary_csv, index=False)

    n_train = (
        int(summary["n_train"].iloc[0])
        if not summary.empty
        else int(args.train_frac * len(y))
    )
    n_test = (
        int(summary["n_test"].iloc[0])
        if not summary.empty
        else len(y) - n_train
    )

    title_metric = (
        "Balanced test error"
        if args.metric == "balanced_error"
        else "Test misclassification rate"
    )

    title = rf"{title_metric} vs. privacy budget on credit-default data"

    plot_error_summary(
        summary,
        out_png=plot_png,
        metric=args.metric,
        shade=args.shade,
        title=title,
    )

    settings = {
        "n_total": len(y),
        "n_train_per_rep": n_train,
        "n_test_per_rep": n_test,
        "p_initial": X.shape[1],
        "p_after_selection": int(summary["p"].iloc[0]) if not summary.empty else X.shape[1],
        "K": K,
        "class_0_count_total": int(np.sum(y == 0)),
        "class_1_count_total": int(np.sum(y == 1)),
        "class_1_prop_total": float(np.mean(y == 1)),
        "train_frac": args.train_frac,
        "reps": args.reps,
        "eps_powers": " ".join(map(str, eps_powers)),
        "eps_grid": " ".join(map(str, eps_grid)),
        "x_axis": "log2(epsilon)",
        "delta": args.delta,
        "Rx": args.Rx,
        "c0": args.c0,
        "eps_split_counts": args.eps_split_counts,
        "feature_mode": args.feature_mode,
        "standardize": args.standardize,
        "select_k": args.select_k,
        "selection_method": args.selection_method,
        "metric": args.metric,
        "prediction_prior": args.prediction_prior,
        "errorbar": args.shade,
        "seed_base": args.seed_base,
    }

    pd.DataFrame([settings]).to_csv(
        outdir / "credit_default_experiment_settings.csv",
        index=False,
    )

    print("========== Credit-default ePTR Bayes experiment ==========")

    for k, v in settings.items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print(f"  raw errors          : {raw_csv}")
    print(f"  error summary       : {summary_csv}")
    print(f"  ePTR components raw : {diag_csv}")
    print(f"  ePTR components sum : {diag_summary_csv}")
    print(f"  selected raw        : {selected_features_raw_csv}")
    print(f"  selected summary    : {selected_features_summary_csv}")
    print(f"  plot                : {plot_png}")
    print(f"  feature names       : {outdir / 'credit_default_feature_names.csv'}")


if __name__ == "__main__":
    main()