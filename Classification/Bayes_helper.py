from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bayes_data import clip_to_ball


@dataclass
class EptrBayesResult:
    released: bool
    mu_tilde: np.ndarray
    means_tilde: np.ndarray
    mu_hat: np.ndarray
    means_hat: np.ndarray
    counts_hat: np.ndarray
    gamma: float
    alpha: float
    M: float
    p_release: float
    noise_sd: float


@dataclass
class WI13BayesResult:
    mu_tilde: np.ndarray
    means_tilde: np.ndarray
    counts_tilde: np.ndarray
    mu_hat: np.ndarray
    means_hat: np.ndarray
    counts_hat: np.ndarray
    count_lap_scale: float
    mean_lap_scales: np.ndarray
    eps_count: float
    eps_mean_total: float


@dataclass
class CauchyBayesResult:
    mu_tilde: np.ndarray
    means_tilde: np.ndarray
    counts_tilde: np.ndarray
    mu_hat: np.ndarray
    means_hat: np.ndarray
    counts_hat: np.ndarray
    count_cauchy_scale: float
    mean_cauchy_scales: np.ndarray
    mean_sensitivities: np.ndarray
    eps_count: float
    eps_mean_total: float


@dataclass
class LDPBayesResult:
    mu_tilde: np.ndarray
    means_tilde: np.ndarray
    mu_raw: np.ndarray
    counts_ldp: np.ndarray
    X_ldp: np.ndarray
    y_ldp: np.ndarray
    G_y: np.ndarray
    q_y: float
    r_y: float
    eps_x: float
    eps_y: float
    eps_x_coord: float
    x_lap_scale: float


def fit_empirical_bayes(X: np.ndarray, y: np.ndarray, K: Optional[int] = None):
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


def sample_cauchy(
    rng: np.random.Generator,
    loc: float = 0.0,
    scale: float = 1.0,
    size=None,
):
    return loc + scale * rng.standard_cauchy(size=size)


def mean_sensitivity_proxy(n_k: int, Rx: float) -> float:
    """Simple sensitivity proxy used by the Cauchy-type baseline."""
    return 2.0 * Rx / (n_k + 1.0)


def fit_bayes_direct_wi13(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
) -> WI13BayesResult:
    """Direct Laplace baseline for private Gaussian naive Bayes."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if Rx <= 0:
        raise ValueError("Rx must be positive.")
    if not (0 < eps_split_counts < 1):
        raise ValueError("eps_split_counts must lie in (0,1).")

    rng = np.random.default_rng(seed)
    n, p = X.shape
    if K is None:
        K = int(np.max(y)) + 1

    X_clip = clip_to_ball(X, Rx)
    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_lap_scale = 2.0 / eps_count
    counts_tilde = counts_hat.astype(float) + rng.laplace(0.0, count_lap_scale, size=K)
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)
    means_tilde = means_hat.copy()
    mean_lap_scales = np.zeros(K, dtype=float)

    for k in range(K):
        sens_mean_k = 2.0 * Rx / (counts_hat[k] + 1.0)
        scale_mean_k = sens_mean_k / eps_per_mean_coord
        mean_lap_scales[k] = scale_mean_k
        means_tilde[k, :] = means_hat[k, :] + rng.laplace(0.0, scale_mean_k, size=p)

    means_tilde = clip_to_ball(means_tilde, Rx)

    return WI13BayesResult(
        mu_tilde=mu_tilde,
        means_tilde=means_tilde,
        counts_tilde=counts_tilde,
        mu_hat=mu_hat,
        means_hat=means_hat,
        counts_hat=counts_hat,
        count_lap_scale=count_lap_scale,
        mean_lap_scales=mean_lap_scales,
        eps_count=eps_count,
        eps_mean_total=eps_mean_total,
    )


def fit_bayes_direct_cauchy(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
) -> CauchyBayesResult:
    """Cauchy-type direct baseline using the same budget split as the Laplace baseline."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if Rx <= 0:
        raise ValueError("Rx must be positive.")
    if not (0 < eps_split_counts < 1):
        raise ValueError("eps_split_counts must lie in (0,1).")

    rng = np.random.default_rng(seed)
    n, p = X.shape
    if K is None:
        K = int(np.max(y)) + 1

    X_clip = clip_to_ball(X, Rx)
    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_cauchy_scale = np.sqrt(2.0) * 2.0 / eps_count
    counts_tilde = counts_hat.astype(float) + sample_cauchy(
        rng=rng,
        loc=0.0,
        scale=count_cauchy_scale,
        size=K,
    )
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)
    means_tilde = means_hat.copy()
    mean_cauchy_scales = np.zeros(K, dtype=float)
    mean_sensitivities = np.zeros(K, dtype=float)

    for k in range(K):
        s_k = mean_sensitivity_proxy(counts_hat[k], Rx)
        mean_sensitivities[k] = s_k
        scale_mean_k = np.sqrt(2.0) * s_k / eps_per_mean_coord
        mean_cauchy_scales[k] = scale_mean_k
        means_tilde[k, :] = means_hat[k, :] + sample_cauchy(
            rng=rng,
            loc=0.0,
            scale=scale_mean_k,
            size=p,
        )

    means_tilde = clip_to_ball(means_tilde, Rx)

    return CauchyBayesResult(
        mu_tilde=mu_tilde,
        means_tilde=means_tilde,
        counts_tilde=counts_tilde,
        mu_hat=mu_hat,
        means_hat=means_hat,
        counts_hat=counts_hat,
        count_cauchy_scale=count_cauchy_scale,
        mean_cauchy_scales=mean_cauchy_scales,
        mean_sensitivities=mean_sensitivities,
        eps_count=eps_count,
        eps_mean_total=eps_mean_total,
    )


def sigmoid(z: float) -> float:
    z = np.clip(z, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def random_fallback_parameter(K: int, p: int, c0: float, Rx: float, rng: np.random.Generator):
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
        theta_tilde = theta_hat + rng.normal(0.0, noise_sd, size=theta_hat.shape)
    else:
        theta_tilde = random_fallback_parameter(K=K, p=p, c0=c0, Rx=Rx, rng=rng)

    return theta_tilde, {
        "released": released,
        "M": float(M),
        "p_release": float(p_release),
        "noise_sd": float(noise_sd),
    }


def fit_bayes_etpr(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    delta: float,
    Rx: float,
    c0: float,
    K: int = 3,
    seed: Optional[int] = None,
) -> EptrBayesResult:
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

    return EptrBayesResult(
        released=info["released"],
        mu_tilde=mu_tilde,
        means_tilde=means_tilde,
        mu_hat=mu_hat,
        means_hat=means_hat,
        counts_hat=counts,
        gamma=gamma,
        alpha=alpha,
        M=info["M"],
        p_release=info["p_release"],
        noise_sd=info["noise_sd"],
    )


def privatize_features_coordwise_laplace(
    X: np.ndarray,
    eps_x: float,
    Rx: float,
    rng: np.random.Generator,
):
    X_clip = clip_to_ball(X, Rx)
    _, p = X_clip.shape

    if np.isinf(eps_x):
        return X_clip.copy(), np.inf, 0.0
    if eps_x <= 0:
        raise ValueError("eps_x must be positive or np.inf.")

    eps_x_coord = eps_x / p
    x_lap_scale = (2.0 * Rx) / eps_x_coord
    X_ldp = X_clip + rng.laplace(0.0, x_lap_scale, size=X_clip.shape)

    return X_ldp, eps_x_coord, x_lap_scale


def privatize_labels_krr(
    y: np.ndarray,
    eps_y: float,
    K: int,
    rng: np.random.Generator,
):
    y = np.asarray(y, dtype=int)

    if np.isinf(eps_y):
        G_y = np.eye(K)
        return y.copy(), G_y, 1.0, 0.0
    if eps_y <= 0:
        raise ValueError("eps_y must be positive or np.inf.")

    ee = np.exp(np.clip(eps_y, 0.0, 60.0))
    q_y = ee / (ee + K - 1.0)
    r_y = 1.0 / (ee + K - 1.0)

    G_y = np.full((K, K), r_y, dtype=float)
    np.fill_diagonal(G_y, q_y)

    y_ldp = np.empty_like(y)
    for i in range(len(y)):
        y_ldp[i] = rng.choice(K, p=G_y[y[i]])

    return y_ldp, G_y, float(q_y), float(r_y)


def fit_bayes_ldp(
    X: np.ndarray,
    y: np.ndarray,
    eps_x: float,
    eps_y: float,
    Rx: float,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    min_mu: float = 1e-6,
) -> LDPBayesResult:
    rng = np.random.default_rng(seed)

    n, p = X.shape
    if K is None:
        K = int(np.max(y)) + 1

    X_ldp, eps_x_coord, x_lap_scale = privatize_features_coordwise_laplace(
        X=X,
        eps_x=eps_x,
        Rx=Rx,
        rng=rng,
    )
    y_ldp, G_y, q_y, r_y = privatize_labels_krr(y=y, eps_y=eps_y, K=K, rng=rng)

    counts_ldp = np.bincount(y_ldp, minlength=K)
    pi_obs = counts_ldp / n

    GinvT = np.linalg.pinv(G_y.T)
    mu_raw = GinvT @ pi_obs

    mu_tilde = np.maximum(mu_raw, min_mu)
    mu_tilde = mu_tilde / mu_tilde.sum()

    T_obs = np.zeros((K, p), dtype=float)
    for b in range(K):
        idx = y_ldp == b
        if np.any(idx):
            T_obs[b] = X_ldp[idx].sum(axis=0) / n

    M_raw = GinvT @ T_obs

    means_tilde = np.zeros((K, p), dtype=float)
    for k in range(K):
        means_tilde[k] = M_raw[k] / max(mu_tilde[k], min_mu)

    means_tilde = clip_to_ball(means_tilde, Rx)

    return LDPBayesResult(
        mu_tilde=mu_tilde,
        means_tilde=means_tilde,
        mu_raw=mu_raw,
        counts_ldp=counts_ldp,
        X_ldp=X_ldp,
        y_ldp=y_ldp,
        G_y=G_y,
        q_y=q_y,
        r_y=r_y,
        eps_x=eps_x,
        eps_y=eps_y,
        eps_x_coord=eps_x_coord,
        x_lap_scale=x_lap_scale,
    )


def predict_naive_bayes(X: np.ndarray, mu: np.ndarray, means: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    means = np.asarray(means, dtype=float)

    if np.any(mu <= 0):
        raise ValueError("All entries of mu must be positive.")

    scores = X @ means.T - 0.5 * np.sum(means**2, axis=1) + np.log(mu)
    return np.argmax(scores, axis=1)


def misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def balanced_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: Optional[int] = None,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if K is None:
        K = int(max(np.max(y_true), np.max(y_pred))) + 1

    class_errors = []
    for k in range(K):
        idx = y_true == k
        if np.any(idx):
            class_errors.append(np.mean(y_pred[idx] != y_true[idx]))

    if not class_errors:
        raise ValueError("No classes found when computing balanced error rate.")

    return float(np.mean(class_errors))


def compute_test_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    K: Optional[int] = None,
) -> float:
    if metric == "misclassification":
        return misclassification_rate(y_true, y_pred)
    if metric == "balanced_error":
        return balanced_error_rate(y_true, y_pred, K=K)
    raise ValueError(f"Unknown metric: {metric}")
