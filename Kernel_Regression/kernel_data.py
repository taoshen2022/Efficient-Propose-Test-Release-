"""Data generation for the 1D kernel-regression simulation in PTR_simu.pdf.

Simulation model
----------------
Y_i = f(X_i) + eps_i,
with eps_i ~ N(0, sigma_eps^2),

f(x) = sin(2*pi*x) + 0.5*cos(4*pi*x),   x in [0, 1].

The simulation section considers two design choices for X:
1) X ~ Unif[0,1] for the epsilon-vs-error curve.
2) X ~ Beta(a,a) for the local-density stress test.

This file only handles data generation. Estimation / ePTR lives in
`eptr_kernel_nw.py`.

Example
-------
python kernel_data.py --n 1000 --design uniform --seed 123
python kernel_data.py --n 1000 --design beta --a 2 --seed 123
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class KernelSimConfig:
    n: int = 1000
    sigma_eps: float = 0.2
    x0: float = 0.5
    design: str = "uniform"  # "uniform" or "beta"
    a: float = 1.0            # only used when design == "beta"
    seed: Optional[int] = None


def true_regression_function(x: np.ndarray | float) -> np.ndarray | float:
    """True regression function f(x)."""
    x_arr = np.asarray(x)
    out = np.sin(2.0 * np.pi * x_arr) + 0.5 * np.cos(4.0 * np.pi * x_arr)
    return float(out) if np.isscalar(x) else out


f_true = true_regression_function


def sample_design(
    n: int,
    design: str = "uniform",
    a: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample X values in [0,1]."""
    if rng is None:
        rng = np.random.default_rng()

    design = design.lower()
    if design == "uniform":
        x = rng.uniform(0.0, 1.0, size=n)
    elif design == "beta":
        if a <= 0:
            raise ValueError("For Beta(a,a), `a` must be positive.")
        x = rng.beta(a, a, size=n)
    else:
        raise ValueError("`design` must be either 'uniform' or 'beta'.")

    return x.astype(float)



def generate_kernel_regression_data(
    config: KernelSimConfig,
) -> dict[str, np.ndarray | float | dict]:
    """Generate one training dataset for the kernel-regression simulation."""
    rng = np.random.default_rng(config.seed)

    x = sample_design(config.n, design=config.design, a=config.a, rng=rng)
    signal = true_regression_function(x)
    eps = rng.normal(loc=0.0, scale=config.sigma_eps, size=config.n)
    y = signal + eps

    out = {
        "x": x,
        "y": y,
        "signal": signal,
        "x0": float(config.x0),
        "f_x0": float(true_regression_function(config.x0)),
        "config": asdict(config),
    }
    return out



def generate_multiple_datasets(
    n_rep: int,
    config: KernelSimConfig,
) -> list[dict[str, np.ndarray | float | dict]]:
    """Generate multiple independent replications.

    Seeds are derived from the base seed if one is provided.
    """
    base_rng = np.random.default_rng(config.seed)
    seeds = base_rng.integers(0, 2**32 - 1, size=n_rep, dtype=np.uint32)

    out = []
    for s in seeds:
        cfg = KernelSimConfig(**asdict(config))
        cfg.seed = int(s)
        out.append(generate_kernel_regression_data(cfg))
    return out



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 1D kernel-regression simulation data.")
    parser.add_argument("--n", type=int, default=1000, help="Sample size.")
    parser.add_argument("--sigma-eps", type=float, default=0.2, help="Noise std deviation.")
    parser.add_argument("--x0", type=float, default=0.5, help="Query point.")
    parser.add_argument(
        "--design",
        type=str,
        default="uniform",
        choices=["uniform", "beta"],
        help="Design distribution for X.",
    )
    parser.add_argument("--a", type=float, default=1.0, help="Beta shape parameter for X ~ Beta(a,a).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = KernelSimConfig(
        n=args.n,
        sigma_eps=args.sigma_eps,
        x0=args.x0,
        design=args.design,
        a=args.a,
        seed=args.seed,
    )
    data = generate_kernel_regression_data(cfg)

    print("Generated dataset")
    print(f"n = {cfg.n}")
    print(f"design = {cfg.design}")
    if cfg.design == "beta":
        print(f"a = {cfg.a}")
    print(f"sigma_eps = {cfg.sigma_eps}")
    print(f"x0 = {cfg.x0}")
    print(f"true f(x0) = {data['f_x0']:.6f}")
    print(f"x range = [{np.min(data['x']):.4f}, {np.max(data['x']):.4f}]")
    print(f"y mean/std = {np.mean(data['y']):.4f} / {np.std(data['y'], ddof=1):.4f}")
