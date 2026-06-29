
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np

from kernel_data import KernelSimConfig, generate_kernel_regression_data


ArrayLike = np.ndarray
NullMode = Literal["nan", "zero", "uniform"]


def clip_to_interval(x: ArrayLike | float, lower: float, upper: float) -> ArrayLike | float:
    """Clip a scalar or array to [lower, upper]."""
    out = np.clip(x, lower, upper)
    return float(out) if np.isscalar(x) else out



def project_to_unit_interval(x: ArrayLike | float) -> ArrayLike | float:
    """Project x to [0,1], the design domain M."""
    return clip_to_interval(x, 0.0, 1.0)



def gaussian_kernel_1d(x0: float, x: ArrayLike, bandwidth: float) -> ArrayLike:
    """Standard 1D Gaussian kernel with bandwidth h.

    K_h(x0, x) = 1/(sqrt(2*pi) h) * exp(-(x0-x)^2 / (2 h^2))
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    z = (x0 - x) / bandwidth
    return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * bandwidth)



def gaussian_kernel_upper_bound_1d() -> float:
    """CK for the 1D Gaussian kernel: sup_x K_h(x0,x) <= CK / h."""
    return 1.0 / math.sqrt(2.0 * math.pi)



def bandwidth_nw(n: int, c_bw: float = 1.0) -> float:
    """Bandwidth h = c * n^{-1/5}."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if c_bw <= 0:
        raise ValueError("c_bw must be positive.")
    return float(c_bw * n ** (-1.0 / 5.0))



def degree_function(x: ArrayLike, x0: float, bandwidth: float) -> float:
    """d_h(x0, X) = sum_i K_h(x0, x_i)."""
    weights = gaussian_kernel_1d(x0, x, bandwidth)
    return float(np.sum(weights))



def nw_estimate(x: ArrayLike, y: ArrayLike, x0: float, bandwidth: float, rf: Optional[float] = None) -> float:
    """Projected nonprivate Nadaraya-Watson estimator at x0."""
    weights = gaussian_kernel_1d(x0, x, bandwidth)
    denom = float(np.sum(weights))
    if denom <= 0:
        raise ZeroDivisionError("Kernel denominator is nonpositive; cannot form NW estimator.")

    theta_hat = float(np.dot(weights, y) / denom)
    if rf is not None:
        theta_hat = float(clip_to_interval(theta_hat, -rf, rf))
    return theta_hat


@dataclass
class EPTRKernelConfig:
    eps: float = 1.0
    delta: Optional[float] = None  # if None, use n^{-3}
    x0: float = 0.5
    c_bw: float = 1.0
    rf: float = 3.0
    c0: float = 0.2
    null_mode: NullMode = "nan"
    seed: Optional[int] = None



def eptr_release_probability(gamma: float, eps: float, delta: float) -> tuple[float, float]:
    """Return (p_release, M) using Algorithm 2."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1).")

    M = 1.0 + (2.0 / eps) * math.log(max(1.0 / delta, 1.0 / eps))
    logit = 0.5 * eps * (gamma - M)

    # numerically stable logistic
    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        p = exp_logit / (1.0 + exp_logit)
    return float(p), float(M)



def _null_numeric_output(rf: float, mode: NullMode, rng: np.random.Generator) -> float:
    if mode == "nan":
        return float("nan")
    if mode == "zero":
        return 0.0
    if mode == "uniform":
        return float(rng.uniform(-rf, rf))
    raise ValueError("Unsupported null_mode.")



def eptr_nw_gaussian_kernel(
    x: ArrayLike,
    y: ArrayLike,
    config: EPTRKernelConfig,
) -> dict[str, float | bool | str]:
    """Kernel-regression ePTR for the 1D Gaussian-kernel NW estimator.

    Returns a dictionary with the nonprivate estimate, private estimate, release
    status, and all intermediate ePTR quantities.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of the same length.")

    n = len(x)
    delta = config.delta if config.delta is not None else n ** (-3)
    rng = np.random.default_rng(config.seed)

    # Step 1 of Algorithm 5: project x to M=[0,1], y to [-Rf, Rf]
    x_proj = project_to_unit_interval(x)
    y_proj = clip_to_interval(y, -config.rf, config.rf)

    # Bandwidth h = c * n^{-1/5}
    h = bandwidth_nw(n, c_bw=config.c_bw)
    ck = gaussian_kernel_upper_bound_1d()

    # Step 2: projected nonprivate kernel regression
    theta_hat = nw_estimate(x_proj, y_proj, x0=config.x0, bandwidth=h, rf=config.rf)
    d_h = degree_function(x_proj, x0=config.x0, bandwidth=h)

    # Step 3: kernel-specific alpha and gamma from Algorithm 5 / Section 5.1
    alpha = 4.0 * config.rf * ck / (h * config.c0 * n)
    gamma = max(d_h - config.c0 * n - 2.0 * ck / h, 0.0) / (2.0 * ck / h)

    p_release, M = eptr_release_probability(gamma=gamma, eps=config.eps, delta=delta)
    release = bool(rng.uniform() < p_release)

    if release:
        noise_sd = (2.0 * alpha / config.eps) * math.sqrt(2.0 * math.log(1.25 / delta))
        theta_priv = float(theta_hat + rng.normal(loc=0.0, scale=noise_sd))
        theta_priv = float(clip_to_interval(theta_priv, -config.rf, config.rf))
        output_type = "released"
    else:
        noise_sd = (2.0 * alpha / config.eps) * math.sqrt(2.0 * math.log(1.25 / delta))
        theta_priv = _null_numeric_output(config.rf, config.null_mode, rng)
        output_type = "null"

    return {
        "n": float(n),
        "x0": float(config.x0),
        "bandwidth": float(h),
        "CK": float(ck),
        "d_h": float(d_h),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "M": float(M),
        "p_release": float(p_release),
        "released": release,
        "noise_sd": float(noise_sd),
        "theta_hat": float(theta_hat),
        "theta_priv": float(theta_priv),
        "rf": float(config.rf),
        "c0": float(config.c0),
        "eps": float(config.eps),
        "delta": float(delta),
        "output_type": output_type,
        "null_mode": config.null_mode,
    }



def gs_nw_gaussian_kernel(
    x: ArrayLike,
    y: ArrayLike,
    eps: float,
    delta: Optional[float] = None,
    x0: float = 0.5,
    c_bw: float = 1.0,
    rf: float = 3.0,
    global_denominator_lb: float = 1.0,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """A simple direct Gaussian baseline (GS-NW).

    This is *not* claimed to be the sharpest implementation. It uses the same
    local-sensitivity-style bound as a conservative global bound by replacing
    d_h(x0, X') with a fixed lower bound `global_denominator_lb`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    delta = delta if delta is not None else n ** (-3)

    x_proj = project_to_unit_interval(x)
    y_proj = clip_to_interval(y, -rf, rf)
    h = bandwidth_nw(n, c_bw=c_bw)
    ck = gaussian_kernel_upper_bound_1d()
    theta_hat = nw_estimate(x_proj, y_proj, x0=x0, bandwidth=h, rf=rf)

    if global_denominator_lb <= 0:
        raise ValueError("global_denominator_lb must be positive.")

    gs_bound = 4.0 * rf * ck / (h * global_denominator_lb)
    noise_sd = (2.0 * gs_bound / eps) * math.sqrt(2.0 * math.log(1.25 / delta))
    rng = np.random.default_rng(seed)
    theta_priv = float(theta_hat + rng.normal(loc=0.0, scale=noise_sd))
    theta_priv = float(clip_to_interval(theta_priv, -rf, rf))

    return {
        "theta_hat": float(theta_hat),
        "theta_priv": float(theta_priv),
        "noise_sd": float(noise_sd),
        "bandwidth": float(h),
        "eps": float(eps),
        "delta": float(delta),
    }

@dataclass
class WaveletNRDPConfig:
    eps: float = 1.0
    delta: Optional[float] = None   # kept only for API compatibility; Laplace gives (eps,0)-DP
    x0: float = 0.5
    rf: float = 1.0                 # clip Y and final estimate to [-rf, rf]
    tau: Optional[float] = None     # if None, use rf
    L: Optional[int] = None         # if None, pick from a simple rule using nu
    nu: float = 1.0                 # nu = alpha - 1/p, used only if L is None
    seed: Optional[int] = None


def haar_father(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = ((x_arr >= 0.0) & (x_arr < 1.0)).astype(float)
    return float(out) if np.isscalar(x) else out


def haar_wavelet_lk(x: np.ndarray | float, l: int, k: int) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    z = (2 ** l) * x_arr - k
    out = np.zeros_like(x_arr, dtype=float)
    amp = 2.0 ** (l / 2.0)
    out[(z >= 0.0) & (z < 0.5)] = amp
    out[(z >= 0.5) & (z < 1.0)] = -amp
    return float(out) if np.isscalar(x) else out


def choose_wavelet_level(n: int, eps: float, nu: float) -> int:
    """
    One-server version of the paper's pointwise choice:
        D^(2 nu + 2) = (n^2 eps^2) ^ (1/(2 nu + 2))  OR  n^(1/(2 nu + 1))
    so D ~ min((n^2 eps^2)^{1/(2 nu + 2)}, n^{1/(2 nu + 1)}), and L ~ log2 D.
    """
    if n <= 1:
        return 0
    D1 = (n * n * eps * eps) ** (1.0 / (2.0 * nu + 2.0))
    D2 = n ** (1.0 / (2.0 * nu + 1.0))
    D = max(1.0, min(D1, D2))
    return max(0, int(math.floor(math.log2(D))))


def wavelet_point_estimate_haar(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    L: int,
    tau: float,
    rf: Optional[float] = None,
) -> float:
    """
    Truncated Haar-series estimator of f(x0):
        a0 * phi(x0) + sum_{l=0}^L b_{lk(x0)} psi_{lk(x0)}(x0),
    where coefficients are empirical averages of clipped Y times basis functions.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_proj = np.clip(x, 0.0, 1.0)
    y_clip = np.clip(y, -tau, tau)

    n = len(x_proj)

    # Father coefficient
    a0 = float(np.mean(y_clip * haar_father(x_proj)))
    theta_hat = a0 * float(haar_father(x0))

    # Only one Haar wavelet per level is active at x0 (except x0=1 edge case)
    if not (0.0 <= x0 < 1.0):
        if rf is not None:
            theta_hat = float(np.clip(theta_hat, -rf, rf))
        return theta_hat

    for l in range(L + 1):
        k0 = min(int(math.floor((2 ** l) * x0)), 2 ** l - 1)
        psi_x = haar_wavelet_lk(x_proj, l, k0)
        psi_x0 = float(haar_wavelet_lk(x0, l, k0))
        b_lk = float(np.mean(y_clip * psi_x))
        theta_hat += b_lk * psi_x0

    if rf is not None:
        theta_hat = float(np.clip(theta_hat, -rf, rf))
    return theta_hat


def wavelet_based_nrdp(
    x: np.ndarray,
    y: np.ndarray,
    config: WaveletNRDPConfig,
) -> dict[str, float | str]:
    """
    Wavelet-based NRDP baseline for point estimation at x0.

    This follows the paper's pointwise template:
      1) clip Y,
      2) form a truncated wavelet estimator of f(x0),
      3) add Laplace noise calibrated to pointwise sensitivity.

    Returns an (eps,0)-DP estimate, hence also (eps,delta)-DP for any delta >= 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    tau = config.rf if config.tau is None else config.tau
    L = choose_wavelet_level(n=n, eps=config.eps, nu=config.nu) if config.L is None else config.L

    theta_hat = wavelet_point_estimate_haar(
        x=x,
        y=y,
        x0=config.x0,
        L=L,
        tau=tau,
        rf=config.rf,
    )

    # Conservative pointwise sensitivity bound for truncated Haar series:
    # changing one record changes each active coefficient by at most 2*tau/n * ||psi||_inf,
    # and after evaluating at x0, summing levels gives O(tau * 2^L / n).
    # We use a safe constant:
    sensitivity = 6.0 * tau * (2 ** L) / n

    rng = np.random.default_rng(config.seed)
    lap_scale = sensitivity / config.eps
    noise = rng.laplace(loc=0.0, scale=lap_scale)

    theta_priv = float(theta_hat + noise)
    theta_priv = float(np.clip(theta_priv, -config.rf, config.rf))

    return {
        "theta_hat": float(theta_hat),
        "theta_priv": float(theta_priv),
        "lap_scale": float(lap_scale),
        "sensitivity": float(sensitivity),
        "L": float(L),
        "tau": float(tau),
        "eps": float(config.eps),
        "delta": float(0.0 if config.delta is None else config.delta),
        "x0": float(config.x0),
        "method": "wavelet-based nrdp",
    }

def pointwise_squared_error(theta: float, truth: float) -> float:
    return float((theta - truth) ** 2)




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gaussian-kernel NW with ePTR.")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument(
        "--delta",
        type=str,
        default="auto",
        help="Use 'auto' for n^{-3}, or provide a numeric value.",
    )
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--c-bw", type=float, default=0.2, help="Bandwidth constant c in h = c * n^{-1/5}.")
    parser.add_argument("--rf", type=float, default=1.0, help="Clipping bound for y and estimator.")
    parser.add_argument("--c0", type=float, default=0.4, help="ePTR design constant for the good set G.")
    parser.add_argument(
        "--null-mode",
        type=str,
        default="nan",
        choices=["nan", "zero", "uniform"],
        help="Numeric representation of the null output.",
    )
    parser.add_argument(
        "--design",
        type=str,
        default="uniform",
        choices=["uniform", "beta"],
        help="Design distribution for X.",
    )
    parser.add_argument("--a", type=float, default=1.0, help="Beta(a,a) parameter when design='beta'.")
    parser.add_argument("--sigma-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    delta = None if args.delta == "auto" else float(args.delta)

    data_cfg = KernelSimConfig(
        n=args.n,
        sigma_eps=args.sigma_eps,
        x0=args.x0,
        design=args.design,
        a=args.a,
        seed=args.seed,
    )
    data = generate_kernel_regression_data(data_cfg)

    est_cfg = EPTRKernelConfig(
        eps=args.eps,
        delta=delta,
        x0=args.x0,
        c_bw=args.c_bw,
        rf=args.rf,
        c0=args.c0,
        null_mode=args.null_mode,
        seed=args.seed + 1,
    )

    out = eptr_nw_gaussian_kernel(data["x"], data["y"], est_cfg)
    truth = float(data["f_x0"])

    print("Kernel-regression ePTR result")
    print(f"true f(x0)   = {truth:.6f}")
    print(f"theta_hat    = {out['theta_hat']:.6f}")
    if math.isnan(out["theta_priv"]):
        priv_str = "nan"
    else:
        priv_str = f"{out['theta_priv']:.6f}"
    print(f"theta_priv   = {priv_str}")
    print(f"released     = {out['released']}")
    print(f"p_release    = {out['p_release']:.6f}")
    print(f"gamma        = {out['gamma']:.6f}")
    print(f"alpha        = {out['alpha']:.6f}")
    print(f"noise_sd     = {out['noise_sd']:.6f}")
    print(f"bandwidth    = {out['bandwidth']:.6f}")
    print(f"d_h          = {out['d_h']:.6f}")
    if out["released"]:
        se = pointwise_squared_error(float(out["theta_priv"]), truth)
        print(f"pointwise SE = {se:.6f}")

    wave_cfg = WaveletNRDPConfig(
        eps=args.eps,
        delta=delta,
        x0=args.x0,
        rf=args.rf,
        nu=1.0,
        L=6,
        seed=args.seed + 2,
    )
    wave_out = wavelet_based_nrdp(data["x"], data["y"], wave_cfg)

    print("\nWavelet-based NRDP result")
    print(f"theta_hat    = {wave_out['theta_hat']:.6f}")
    print(f"theta_priv   = {wave_out['theta_priv']:.6f}")
    print(f"L            = {int(wave_out['L'])}")
    print(f"lap_scale    = {wave_out['lap_scale']:.6f}")
