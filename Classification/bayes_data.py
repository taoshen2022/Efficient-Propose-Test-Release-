import numpy as np
from typing import Optional, Sequence


def make_class_means(
    K: int,
    delta: float = 2.0,
    p: int = 10,
) -> np.ndarray:
    """
    Construct class means for a K-class Gaussian Bayes model.

    Generalization of your 3-class design:
      class 0 -> (+delta, 0, ..., 0)
      class 1 -> (-delta, 0, ..., 0)
      class 2 -> (0, +delta, 0, ..., 0)
      class 3 -> (0, -delta, 0, ..., 0)
      class 4 -> (0, 0, +delta, ..., 0)
      class 5 -> (0, 0, -delta, ..., 0)
      ...

    So each pair of classes uses one coordinate axis with opposite signs.

    Parameters
    ----------
    K : int
        Number of classes.
    delta : float
        Mean separation magnitude.
    p : int
        Feature dimension. Must satisfy p >= ceil(K / 2).

    Returns
    -------
    means : ndarray of shape (K, p)
    """
    if K <= 0:
        raise ValueError("K must be positive.")

    min_required_p = (K + 1) // 2
    if p < min_required_p:
        raise ValueError(
            f"p must be at least ceil(K/2) = {min_required_p} for this construction."
        )

    means = np.zeros((K, p), dtype=float)

    for k in range(K):
        axis = k // 2
        sign = 1.0 if (k % 2 == 0) else -1.0
        means[k, axis] = sign * delta

    return means


def clip_to_ball(X: np.ndarray, radius: float) -> np.ndarray:
    """
    Project each row of X onto the Euclidean ball of radius 'radius'.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return X * scale


def generate_bayes_data(
    n: int,
    K: Optional[int] = None,
    mu: Optional[Sequence[float]] = None,
    delta: float = 2.0,
    p: int = 10,
    Rx: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Generate data from the Gaussian Bayes model:
        P(Y = k) = mu_k
        X | Y = k ~ N(m_k, I_p),   k = 0, ..., K-1

    Parameters
    ----------
    n : int
        Sample size.
    K : int or None
        Number of classes. Required if mu is None.
    mu : sequence of length K or None
        Class probabilities. If None, use uniform probabilities over K classes.
    delta : float
        Mean separation parameter.
    p : int
        Feature dimension.
    Rx : float or None
        If not None, clip each feature vector to radius Rx.
    seed : int or None
        Random seed.

    Returns
    -------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
        Labels in {0, 1, ..., K-1}.
    true_mu : ndarray, shape (K,)
    true_means : ndarray, shape (K, p)
    """
    rng = np.random.default_rng(seed)

    if mu is None:
        if K is None:
            raise ValueError("At least one of K or mu must be provided.")
        if K <= 0:
            raise ValueError("K must be positive.")
        mu = np.ones(K, dtype=float) / K
    else:
        mu = np.asarray(mu, dtype=float)
        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array-like object.")
        if np.any(mu <= 0) or not np.isclose(mu.sum(), 1.0):
            raise ValueError("mu must be positive and sum to 1.")
        if K is None:
            K = len(mu)
        elif len(mu) != K:
            raise ValueError("If both K and mu are provided, len(mu) must equal K.")

    true_means = make_class_means(K=K, delta=delta, p=p)

    y = rng.choice(K, size=n, p=mu)
    X = rng.normal(size=(n, p)) + true_means[y]

    if Rx is not None:
        X = clip_to_ball(X, Rx)

    return X, y, mu, true_means