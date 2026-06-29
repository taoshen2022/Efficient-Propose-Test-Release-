from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from linear_data import LinearSimConfig, generate_linear_regression_data


@dataclass
class DPFMLinearConfig:
    epsilon: float = 1.0
    Rx: float = 3.0          # row-wise l2 clipping bound for x_i in the original scale
    Ry: float = 9.0          # clipping bound for y_i in the original scale
    C: Optional[float] = None  # optional l2 projection radius for beta after optimization
    # Kept for backward compatibility with existing experiment scripts.
    # The rewritten implementation uses Zhang et al.'s regularization heuristic
    # and spectral trimming instead of the old min-eigenvalue floor routine.
    min_eig_floor: float = 1e-6
    jitter: float = 1e-8
    # Optional explicit paper-style regularization level. If None, use
    # lambda = 4 * sd(Laplace(scale = Delta / epsilon)).
    lambda_reg: Optional[float] = None
    use_spectral_trimming: bool = True
    seed: Optional[int] = None


# ---------- utilities ----------

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



def clip_vector_l2(v: np.ndarray, radius: float | None) -> np.ndarray:
    if radius is None:
        return v.copy()
    if radius <= 0:
        raise ValueError("radius must be positive")
    norm = float(np.linalg.norm(v))
    if norm <= radius:
        return v.copy()
    return v * (radius / max(norm, 1e-12))



def prediction_mse(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    return float(np.mean((y_test - X_test @ beta) ** 2))



def parameter_sq_error(beta: np.ndarray, beta_true: np.ndarray) -> float:
    return float(np.sum((beta - beta_true) ** 2))


# ---------- Functional Mechanism for linear regression ----------

def polynomial_sensitivity_bound(p: int) -> float:
    """Coarse sensitivity bound from Zhang et al. (2012) for linear regression.

    After normalizing data so that ||x_i||_2 <= 1 and |y_i| <= 1, Section 4.2 gives
    Delta <= 2 (p + 1)^2 for the l1 sensitivity of the vector of polynomial
    coefficients of the quadratic objective.
    """
    return float(2.0 * (p + 1) ** 2)



def quadratic_coefficients_linear_regression(
    X_unit: np.ndarray,
    y_unit: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Exact polynomial coefficients of the normalized least-squares objective.

    For normalized data, the objective is
        f(theta) = sum_i (y_i - x_i^T theta)^2
                 = const + linear^T theta + theta^T M theta,
    where
        const  = sum_i y_i^2,
        linear = -2 X^T y,
        M      = X^T X.
    """
    M = X_unit.T @ X_unit
    b = X_unit.T @ y_unit
    const = float(np.sum(y_unit ** 2))
    linear = -2.0 * b
    return {
        "const": const,
        "linear": linear,
        "M": M,
        "b": b,
    }



def privatize_quadratic_objective(
    X_unit: np.ndarray,
    y_unit: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Paper-style FM perturbation of quadratic objective coefficients.

    This matches the linear-regression specialization in Zhang et al. (2012):
    - use Delta = 2 (p + 1)^2,
    - add Laplace(Delta / epsilon) noise to each polynomial coefficient,
    - add noise only to the upper-triangular part of the quadratic matrix and
      mirror it to preserve symmetry.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    coeffs = quadratic_coefficients_linear_regression(X_unit, y_unit)
    M = np.asarray(coeffs["M"], dtype=float)
    linear = np.asarray(coeffs["linear"], dtype=float)
    const = float(coeffs["const"])
    p = M.shape[0]

    sens = polynomial_sensitivity_bound(p)
    scale = float(sens / epsilon)

    const_noisy = const + float(rng.laplace(loc=0.0, scale=scale))
    linear_noisy = linear + rng.laplace(loc=0.0, scale=scale, size=linear.shape)

    M_noisy = np.zeros_like(M)
    upper_noise = rng.laplace(loc=0.0, scale=scale, size=(p, p))
    for j in range(p):
        for l in range(j, p):
            M_noisy[j, l] = M[j, l] + upper_noise[j, l]
            M_noisy[l, j] = M_noisy[j, l]

    return {
        "const": const,
        "linear": linear,
        "M": M,
        "b": np.asarray(coeffs["b"], dtype=float),
        "const_noisy": const_noisy,
        "linear_noisy": linear_noisy,
        "M_noisy": M_noisy,
        "laplace_scale": scale,
        "sensitivity": sens,
    }



def default_regularization_lambda(laplace_scale: float) -> float:
    """Paper's heuristic: lambda = 4 * sd(Laplace(scale))."""
    return float(4.0 * np.sqrt(2.0) * laplace_scale)



def regularize_and_trim_hessian(
    M_noisy: np.ndarray,
    *,
    laplace_scale: float,
    lambda_reg: float | None = None,
    use_spectral_trimming: bool = True,
    numerical_fallback_jitter: float = 1e-10,
) -> dict[str, np.ndarray | float | bool]:
    """Apply the paper's regularization idea and spectral trimming.

    The main returned matrix `M_effective` is the PSD matrix used in the final
    optimization. If spectral trimming is enabled, it is obtained by removing
    non-positive eigenvalues from M_reg. A tiny extra diagonal shift is used only
    as a last-resort numerical safeguard when regularization still leaves no
    positive direction at all.
    """
    M_sym = 0.5 * (M_noisy + M_noisy.T)
    eigvals_noisy, eigvecs_noisy = np.linalg.eigh(M_sym)
    min_eig_noisy = float(eigvals_noisy.min())

    lambda_used = default_regularization_lambda(laplace_scale) if lambda_reg is None else float(lambda_reg)
    M_reg = M_sym + lambda_used * np.eye(M_sym.shape[0])

    eigvals_reg, eigvecs_reg = np.linalg.eigh(M_reg)
    positive_mask = eigvals_reg > 0.0
    n_positive = int(np.sum(positive_mask))
    spectral_trimmed = bool(use_spectral_trimming and n_positive < M_reg.shape[0])
    numerical_fallback = False

    if use_spectral_trimming:
        if n_positive == 0:
            # Rare numerical safety fallback; not part of the paper, but avoids a
            # completely ill-posed benchmark if all eigenvalues remain non-positive.
            shift = float(-eigvals_reg.max() + numerical_fallback_jitter)
            M_reg = M_reg + shift * np.eye(M_reg.shape[0])
            eigvals_reg, eigvecs_reg = np.linalg.eigh(M_reg)
            positive_mask = eigvals_reg > 0.0
            n_positive = int(np.sum(positive_mask))
            numerical_fallback = True
            spectral_trimmed = bool(n_positive < M_reg.shape[0])
        eigvals_effective = eigvals_reg[positive_mask]
        eigvecs_effective = eigvecs_reg[:, positive_mask]
        M_effective = eigvecs_effective @ np.diag(eigvals_effective) @ eigvecs_effective.T
    else:
        eigvals_effective = eigvals_reg
        eigvecs_effective = eigvecs_reg
        M_effective = M_reg

    return {
        "M_sym": M_sym,
        "M_reg": M_reg,
        "M_effective": M_effective,
        "eigvals_noisy": eigvals_noisy,
        "eigvals_reg": eigvals_reg,
        "eigvecs_effective": eigvecs_effective,
        "eigvals_effective": eigvals_effective,
        "eig_min_noisy": min_eig_noisy,
        "lambda_reg": lambda_used,
        "regularized": True,
        "spectral_trimmed": spectral_trimmed,
        "n_positive_eigs": n_positive,
        "numerical_fallback": numerical_fallback,
        # Backward-compatible summary flag used by the current experiment utils.
        "convexified": bool(spectral_trimmed or numerical_fallback or (lambda_used > 0.0)),
    }



def solve_private_quadratic(
    linear_coeff: np.ndarray,
    eigvecs_effective: np.ndarray,
    eigvals_effective: np.ndarray,
) -> np.ndarray:
    """Solve min_theta theta^T M theta + linear^T theta using trimmed eigensystem."""
    if eigvals_effective.size == 0:
        return np.zeros_like(linear_coeff)

    proj = eigvecs_effective.T @ linear_coeff
    v_star = -0.5 * (proj / eigvals_effective)
    theta_star = eigvecs_effective @ v_star
    return theta_star



def fit_dp_fm_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    Rx: float,
    Ry: float,
    rng: np.random.Generator,
    C: Optional[float] = None,
    min_eig_floor: float = 1e-6,  # backward-compatible, unused
    jitter: float = 1e-8,         # backward-compatible, unused except for guard
    lambda_reg: Optional[float] = None,
    use_spectral_trimming: bool = True,
) -> dict[str, np.ndarray | float | bool]:
    """FM benchmark closer to Zhang et al. (2012) for linear regression.

    Steps:
    1. Clip and normalize to ||x_i||_2 <= 1 and |y_i| <= 1.
    2. Build the exact quadratic polynomial coefficients.
    3. Add Laplace noise coefficient-wise using Delta = 2 (p + 1)^2.
    4. Apply paper-style regularization and spectral trimming.
    5. Solve the noisy quadratic objective.
    6. Rescale back to the original parameterization: beta = (Ry / Rx) * theta.
    """
    _ = min_eig_floor  # retained only for backward compatibility with old scripts

    X_clip = clip_rows_l2(X, Rx)
    y_clip = clip_scalar(y, Ry)

    X_unit = X_clip / Rx
    y_unit = y_clip / Ry

    priv = privatize_quadratic_objective(X_unit, y_unit, epsilon=epsilon, rng=rng)
    reg = regularize_and_trim_hessian(
        np.asarray(priv["M_noisy"], dtype=float),
        laplace_scale=float(priv["laplace_scale"]),
        lambda_reg=lambda_reg,
        use_spectral_trimming=use_spectral_trimming,
        numerical_fallback_jitter=max(float(jitter), 1e-12),
    )

    theta_unit_priv = solve_private_quadratic(
        np.asarray(priv["linear_noisy"], dtype=float),
        np.asarray(reg["eigvecs_effective"], dtype=float),
        np.asarray(reg["eigvals_effective"], dtype=float),
    )
    beta_priv = (Ry / Rx) * theta_unit_priv
    beta_priv = clip_vector_l2(beta_priv, C)

    # Nonprivate clipped baseline on the same clipped-and-rescaled data.
    theta_unit_np = np.linalg.pinv(np.asarray(priv["M"], dtype=float)) @ np.asarray(priv["b"], dtype=float)
    beta_np_clip = (Ry / Rx) * theta_unit_np
    beta_np_clip = clip_vector_l2(beta_np_clip, C)

    return {
        **priv,
        **reg,
        "X_clipped": X_clip,
        "y_clipped": y_clip,
        "X_unit": X_unit,
        "y_unit": y_unit,
        "beta_nonprivate_clip": beta_np_clip,
        "beta_private": beta_priv,
    }


# ---------- simulation wrapper ----------

def run_single_demo(sim_cfg: LinearSimConfig, fm_cfg: DPFMLinearConfig) -> dict[str, np.ndarray | float | bool]:
    data = generate_linear_regression_data(sim_cfg)
    rng = np.random.default_rng(fm_cfg.seed)

    fit = fit_dp_fm_linear_regression(
        data["X_train"],
        data["y_train"],
        epsilon=fm_cfg.epsilon,
        Rx=fm_cfg.Rx,
        Ry=fm_cfg.Ry,
        rng=rng,
        C=fm_cfg.C,
        min_eig_floor=fm_cfg.min_eig_floor,
        jitter=fm_cfg.jitter,
        lambda_reg=fm_cfg.lambda_reg,
        use_spectral_trimming=fm_cfg.use_spectral_trimming,
    )

    return {
        **fit,
        "theta_true": data["theta_true"],
        "mse_test_nonprivate_clip": prediction_mse(fit["beta_nonprivate_clip"], data["X_test"], data["y_test"]),
        "mse_test_private": prediction_mse(fit["beta_private"], data["X_test"], data["y_test"]),
        "param_err_nonprivate_clip": parameter_sq_error(fit["beta_nonprivate_clip"], data["theta_true"]),
        "param_err_private": parameter_sq_error(fit["beta_private"], data["theta_true"]),
    }



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Functional Mechanism benchmark for linear regression, rewritten to more closely follow Zhang et al. (2012)."
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sigma_eps", type=float, default=1.0)
    parser.add_argument("--n_test", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--Rx", type=float, default=3.0)
    parser.add_argument("--Ry", type=float, default=9.0)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=None,
        help="Optional paper-style regularization constant. Default: 4 * sd(Laplace(scale = Delta/epsilon)).",
    )
    parser.add_argument(
        "--no_spectral_trimming",
        action="store_true",
        help="Disable spectral trimming after regularization.",
    )
    # Backward-compatible parser entries.
    parser.add_argument("--min_eig_floor", type=float, default=1e-6)
    parser.add_argument("--jitter", type=float, default=1e-8)
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
    fm_cfg = DPFMLinearConfig(
        epsilon=args.epsilon,
        Rx=args.Rx,
        Ry=args.Ry,
        C=args.C,
        min_eig_floor=args.min_eig_floor,
        jitter=args.jitter,
        lambda_reg=args.lambda_reg,
        use_spectral_trimming=not args.no_spectral_trimming,
        seed=args.seed,
    )
    out = run_single_demo(sim_cfg, fm_cfg)
    print(f"laplace scale         : {out['laplace_scale']:.6f}")
    print(f"sensitivity           : {out['sensitivity']:.6f}")
    print(f"eig min noisy Hessian : {out['eig_min_noisy']:.6f}")
    print(f"lambda_reg            : {out['lambda_reg']:.6f}")
    print(f"spectral trimmed      : {out['spectral_trimmed']}")
    print(f"numerical fallback    : {out['numerical_fallback']}")
    print(f"test MSE (clip OLS)   : {out['mse_test_nonprivate_clip']:.6f}")
    print(f"test MSE (FM)         : {out['mse_test_private']:.6f}")
    print(f"param err (clip OLS)  : {out['param_err_nonprivate_clip']:.6f}")
    print(f"param err (FM)        : {out['param_err_private']:.6f}")
