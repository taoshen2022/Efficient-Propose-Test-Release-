from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from linear_data import LinearSimConfig, generate_linear_regression_data


@dataclass
class DPCaiLRConfig:
    epsilon: float = 1.0
    delta: Optional[float] = None
    T: Optional[int] = None
    eta0: Optional[float] = None
    Rx: Optional[float] = None          # row clipping radius for x_i
    R: Optional[float] = None           # response truncation level in Pi_R(y_i)
    C: float = 1.0                      # feasibility radius ||beta||_2 <= C
    beta0: Optional[np.ndarray] = None
    seed: Optional[int] = None


def clip_rows_l2(X: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scales = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return X * scales


def clip_vector_l2(v: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    norm = np.linalg.norm(v)
    if norm <= radius:
        return v.copy()
    return v * (radius / max(norm, 1e-12))


def clip_scalar(y: np.ndarray, bound: float) -> np.ndarray:
    if bound <= 0:
        raise ValueError("bound must be positive")
    return np.clip(y, -bound, bound)


def suggested_Rx(n: int, p: int, tau: float) -> float:
    """Practical default used in earlier scripts: sqrt(trace(Sigma)) + 1."""
    diag = np.ones(p, dtype=float)
    diag[-min(5, p):] = tau
    return float(np.sqrt(diag.sum()) + 1.0)


def projected_ols(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    gram = X.T @ X
    rhs = X.T @ y
    beta = np.linalg.pinv(gram) @ rhs
    return clip_vector_l2(beta, C)


def default_step_size(X: np.ndarray) -> float:
    """Simple stable step size for the empirical squared loss gradient.

    Gradient of L_n(beta) = n^{-1} sum (y_i - x_i^T beta)^2 is
        (2/n) X^T (X beta - y).
    The algorithm in the paper absorbs constants into eta0. We use
    eta0 = 1 / lambda_max((X^T X)/n), which is conservative and works well.
    """
    gram = (X.T @ X) / X.shape[0]
    lam_max = float(np.linalg.eigvalsh(gram).max())
    return 1.0 / max(lam_max, 1e-8)


def noise_scale_per_coordinate(eta0: float, B: float, n: int, epsilon: float, delta: float, T: int) -> float:
    # From Algorithm 4.1: variance = (eta0)^2 * 2 B^2 log(2T/delta) / (n^2 (epsilon/T)^2)
    return (eta0 * B * np.sqrt(2.0 * np.log(2.0 * T / delta))) / (n * (epsilon / T))


def fit_dp_linear_regression_cai2021(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    T: int,
    eta0: float,
    Rx: float,
    R: float,
    C: float,
    rng: np.random.Generator,
    beta0: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | float]:
    """Implement Algorithm 4.1 of Cai-Wang-Zhang (2021) on clipped data.

    Privacy guarantee in the paper assumes bounded design ||x_i|| <= c_x and ||beta|| <= c_0.
    Since the user's simulation has Gaussian design, we first clip rows of X to radius Rx.
    Then the paper's noise condition becomes B >= 4 (R + C*Rx) Rx.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0, 1)")
    if T <= 0:
        raise ValueError("T must be positive")

    n, p = X.shape
    X_clip = clip_rows_l2(X, Rx)
    y_trunc = clip_scalar(y, R)

    # Paper condition with c_x replaced by clipping radius Rx and c_0 by feasibility radius C.
    B = 4.0 * (R + C * Rx) * Rx
    noise_sd = noise_scale_per_coordinate(eta0, B, n, epsilon, delta, T)

    if beta0 is None:
        beta = np.zeros(p, dtype=float)
    else:
        beta = clip_vector_l2(np.asarray(beta0, dtype=float), C)

    traj = [beta.copy()]
    for _ in range(T):
        resid = X_clip @ beta - y_trunc
        grad = (X_clip.T @ resid) / n  # matches the displayed algorithm's scaling
        w = rng.normal(loc=0.0, scale=noise_sd, size=p)
        beta = clip_vector_l2(beta - eta0 * grad + w, C)
        traj.append(beta.copy())

    return {
        "beta_priv": beta,
        "B": B,
        "noise_sd": noise_sd,
        "X_clipped": X_clip,
        "y_trunc": y_trunc,
        "traj": np.asarray(traj),
    }


def prediction_mse(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    resid = y_test - X_test @ beta
    return float(np.mean(resid ** 2))


def parameter_sq_error(beta: np.ndarray, beta_true: np.ndarray) -> float:
    return float(np.sum((beta - beta_true) ** 2))


def run_single_demo(sim_cfg: LinearSimConfig, dp_cfg: DPCaiLRConfig) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(dp_cfg.seed)
    data = generate_linear_regression_data(sim_cfg)

    delta = dp_cfg.delta if dp_cfg.delta is not None else sim_cfg.n ** (-3)
    Rx = dp_cfg.Rx if dp_cfg.Rx is not None else suggested_Rx(sim_cfg.n, sim_cfg.p, sim_cfg.tau)
    # In the user's DGP, ||theta_true||_2 = 1 and sigma_eps = 1; a practical truncation is a few SDs.
    R = dp_cfg.R if dp_cfg.R is not None else 4.0

    X_clip = clip_rows_l2(data["X_train"], Rx)
    eta0 = dp_cfg.eta0 if dp_cfg.eta0 is not None else default_step_size(X_clip)
    T = dp_cfg.T if dp_cfg.T is not None else int(np.ceil(5 * np.log(max(sim_cfg.n, 2))))

    fit = fit_dp_linear_regression_cai2021(
        data["X_train"],
        data["y_train"],
        epsilon=dp_cfg.epsilon,
        delta=delta,
        T=T,
        eta0=eta0,
        Rx=Rx,
        R=R,
        C=dp_cfg.C,
        rng=rng,
        beta0=dp_cfg.beta0,
    )

    beta_ols_clip = projected_ols(X_clip, clip_scalar(data["y_train"], R), dp_cfg.C)

    return {
        **fit,
        "beta_ols_clip": beta_ols_clip,
        "beta_true": data["theta_true"],
        "mse_test_ols_clip": prediction_mse(beta_ols_clip, data["X_test"], data["y_test"]),
        "mse_test_private": prediction_mse(fit["beta_priv"], data["X_test"], data["y_test"]),
        "param_err_ols_clip": parameter_sq_error(beta_ols_clip, data["theta_true"]),
        "param_err_private": parameter_sq_error(fit["beta_priv"], data["theta_true"]),
        "delta": delta,
        "Rx": Rx,
        "R": R,
        "T": T,
        "eta0": eta0,
        "n": sim_cfg.n,
        "p": sim_cfg.p,
        "tau": sim_cfg.tau,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Algorithm 4.1 of Cai-Wang-Zhang (2021) under the user's linear-regression DGP.")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sigma_eps", type=float, default=1.0)
    parser.add_argument("--n_test", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--eta0", type=float, default=None)
    parser.add_argument("--Rx", type=float, default=3.0)
    parser.add_argument("--R", type=float, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()

    sim_cfg = LinearSimConfig(
        n=args.n,
        p=args.p,
        tau=args.tau,
        sigma_eps=args.sigma_eps,
        n_test=args.n_test,
        seed=args.seed,
    )
    dp_cfg = DPCaiLRConfig(
        epsilon=args.epsilon,
        delta=args.delta,
        T=args.T,
        eta0=args.eta0,
        Rx=args.Rx,
        R=args.R,
        C=args.C,
        seed=args.seed,
    )

    out = run_single_demo(sim_cfg, dp_cfg)
    print(f"B                    : {out['B']:.6f}")
    print(f"noise sd             : {out['noise_sd']:.6f}")
    print(f"T                    : {out['T']}")
    print(f"eta0                 : {out['eta0']:.6f}")
    print(f"Rx                   : {out['Rx']:.6f}")
    print(f"R                    : {out['R']:.6f}")
    print(f"delta                : {out['delta']:.6e}")
    print(f"test MSE (OLS clip)  : {out['mse_test_ols_clip']:.6f}")
    print(f"test MSE (DP Alg4.1) : {out['mse_test_private']:.6f}")
    print(f"param err (OLS clip) : {out['param_err_ols_clip']:.6f}")
    print(f"param err (DP Alg4.1): {out['param_err_private']:.6f}")
