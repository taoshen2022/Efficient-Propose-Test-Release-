from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from linear_data import LinearSimConfig, generate_linear_regression_data


@dataclass
class DPOLSAlg1Config:
    epsilon: float = 1.0
    delta: float = 1e-6
    r: Optional[int] = None  # number of JL rows; if None, use min(n, 5p)
    Rx: float = 4.5          # l2 row clipping bound for x_i
    Ry: float = 4.0          # clipping bound for y_i
    seed: Optional[int] = None


# ---------- basic utilities ----------

def clip_rows_l2(X: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scales = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return X * scales


def clip_scalar(y: np.ndarray, bound: float) -> np.ndarray:
    if bound <= 0:
        raise ValueError("bound must be positive")
    return np.clip(y, -bound, bound)


def ols_from_xy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # pinv is safer numerically; in the altered case the problem is ridge-like anyway.
    return np.linalg.pinv(X) @ y


def prediction_mse(theta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    return float(np.mean((y_test - X_test @ theta) ** 2))


def parameter_sq_error(theta: np.ndarray, theta_true: np.ndarray) -> float:
    return float(np.sum((theta - theta_true) ** 2))


# ---------- Algorithm 1 from Sheffet (2017), specialized to A=[X|y] ----------

def stack_A(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([X, y])


def split_A(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return A[:, :-1], A[:, -1]


def algorithm1_private_projection(
    A: np.ndarray,
    *,
    B: float,
    epsilon: float,
    delta: float,
    r: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float | bool]:
    """Implement Algorithm 1 ('private JLT projection') on a bounded-row matrix A.

    A is n x d. The algorithm either releases R A ('unaltered') or R A', where
    A' = [A; w I_d] ('altered').
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0, 1)")
    n, d = A.shape
    if r <= d:
        raise ValueError("r must be strictly larger than d so the sketched OLS problem is not underdetermined by default.")

    log_term = np.log(8.0 / delta)
    w_sq = (8.0 * B**2 / epsilon) * (np.sqrt(2.0 * r * log_term) + 2.0 * log_term)
    w = float(np.sqrt(w_sq))

    # In numpy, laplace(loc, scale) has variance 2*scale^2.
    Z = float(rng.laplace(loc=0.0, scale=4.0 * B**2 / epsilon))

    sigma_min_sq = float(np.linalg.eigvalsh(A.T @ A).min())
    threshold = w_sq + Z + (4.0 * B**2 * np.log(1.0 / delta) / epsilon)

    if sigma_min_sq > threshold:
        R = rng.normal(loc=0.0, scale=1.0, size=(r, n))
        A_proj = R @ A
        altered = False
    else:
        A_prime = np.vstack([A, w * np.eye(d)])
        R = rng.normal(loc=0.0, scale=1.0, size=(r, n + d))
        A_proj = R @ A_prime
        altered = True

    return {
        "A_proj": A_proj,
        "altered": altered,
        "sigma_min_sq": sigma_min_sq,
        "threshold": threshold,
        "w": w,
        "w_sq": w_sq,
        "laplace_noise": Z,
        "B": B,
        "r": r,
    }


def fit_dp_ols_alg1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    r: int,
    Rx: float,
    Ry: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float | bool]:
    """Private OLS estimator obtained by Algorithm 1's sketched release.

    This is an estimation-oriented wrapper around the paper's algorithm.
    We clip x_i and y_i first so every row a_i = (x_i, y_i) has norm at most
    B = sqrt(Rx^2 + Ry^2), build A=[X|y], run Algorithm 1, then solve OLS on
    the released sketch. If the matrix is altered, this corresponds to a
    ridge-like fit induced by the appended w I term.
    """
    X_clip = clip_rows_l2(X, Rx)
    y_clip = clip_scalar(y, Ry)
    B = float(np.sqrt(Rx**2 + Ry**2))
    A = stack_A(X_clip, y_clip)

    proj = algorithm1_private_projection(
        A,
        B=B,
        epsilon=epsilon,
        delta=delta,
        r=r,
        rng=rng,
    )

    X_priv, y_priv = split_A(proj["A_proj"])
    beta_priv = ols_from_xy(X_priv, y_priv)
    beta_np = ols_from_xy(X_clip, y_clip)

    return {
        **proj,
        "X_clipped": X_clip,
        "y_clipped": y_clip,
        "beta_nonprivate": beta_np,
        "beta_private": beta_priv,
    }


# ---------- simulation wrapper for your setting ----------

def default_r(n: int, p: int) -> int:
    # Practical default: modest oversampling above p, capped by n.
    return int(max(p + 10, min(n, 5 * p)))


def run_single_demo(sim_cfg: LinearSimConfig, dp_cfg: DPOLSAlg1Config) -> dict[str, np.ndarray | float | bool]:
    data = generate_linear_regression_data(sim_cfg)
    rng = np.random.default_rng(dp_cfg.seed)
    r = dp_cfg.r if dp_cfg.r is not None else default_r(sim_cfg.n, sim_cfg.p)

    fit = fit_dp_ols_alg1(
        data["X_train"],
        data["y_train"],
        epsilon=dp_cfg.epsilon,
        delta=dp_cfg.delta,
        r=r,
        Rx=dp_cfg.Rx,
        Ry=dp_cfg.Ry,
        rng=rng,
    )

    out = {
        **fit,
        "theta_true": data["theta_true"],
        "mse_test_nonprivate": prediction_mse(fit["beta_nonprivate"], data["X_test"], data["y_test"]),
        "mse_test_private": prediction_mse(fit["beta_private"], data["X_test"], data["y_test"]),
        "param_err_nonprivate": parameter_sq_error(fit["beta_nonprivate"], data["theta_true"]),
        "param_err_private": parameter_sq_error(fit["beta_private"], data["theta_true"]),
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DP OLS via Algorithm 1 of Sheffet (2017), specialized to the simulation setting.")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sigma_eps", type=float, default=1.0)
    parser.add_argument("--n_test", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--Rx", type=float, default=3.0)
    parser.add_argument("--Ry", type=float, default=9.0)
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
    dp_cfg = DPOLSAlg1Config(
        epsilon=args.epsilon,
        delta=args.delta,
        r=args.r,
        Rx=args.Rx,
        Ry=args.Ry,
        seed=args.seed,
    )
    out = run_single_demo(sim_cfg, dp_cfg)
    print(f"altered              : {out['altered']}")
    print(f"r                    : {out['r']}")
    print(f"B                    : {out['B']:.6f}")
    print(f"w                    : {out['w']:.6f}")
    print(f"sigma_min(A)^2       : {out['sigma_min_sq']:.6f}")
    print(f"private threshold    : {out['threshold']:.6f}")
    print(f"test MSE (OLS clip)  : {out['mse_test_nonprivate']:.6f}")
    print(f"test MSE (DP Alg1)   : {out['mse_test_private']:.6f}")
    print(f"param err (OLS clip) : {out['param_err_nonprivate']:.6f}")
    print(f"param err (DP Alg1)  : {out['param_err_private']:.6f}")
