from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from scipy.stats import beta as beta_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernel_data import KernelSimConfig, generate_kernel_regression_data
from eptr_kernel_nw import (
    bandwidth_nw,
    nw_estimate,
    EPTRKernelConfig,
    eptr_nw_gaussian_kernel,
    WaveletNRDPConfig,
    wavelet_based_nrdp,
)


METHOD_ORDER = ["Non-private", "WPE", "ePTR"]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def pointwise_se(theta: float, truth: float) -> float:
    return float((theta - truth) ** 2)


def run_one_rep(
    *,
    n: int,
    eps: float,
    a: float,
    x0: float,
    c_bw: float,
    rf: float,
    c0: float,
    sigma_eps: float,
    null_mode: str,
    wavelet_L: int,
    wavelet_nu: float,
    seed: int,
) -> list[dict]:
    data_cfg = KernelSimConfig(
        n=n,
        sigma_eps=sigma_eps,
        x0=x0,
        design="beta",
        a=a,
        seed=seed,
    )
    data = generate_kernel_regression_data(data_cfg)

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    truth = float(data["f_x0"])

    x_proj = np.clip(x, 0.0, 1.0)
    y_proj = np.clip(y, -rf, rf)

    h = bandwidth_nw(n, c_bw=c_bw)
    theta_np = nw_estimate(x_proj, y_proj, x0=x0, bandwidth=h, rf=rf)

    u = beta_dist.cdf(x, a, a)
    u0 = float(beta_dist.cdf(x0, a, a))

    wave_cfg = WaveletNRDPConfig(
        eps=eps,
        delta=None,
        x0=u0,
        rf=rf,
        tau=rf,
        L=wavelet_L,
        nu=wavelet_nu,
        seed=seed + 11,
    )
    wave_out = wavelet_based_nrdp(u, y, wave_cfg)

    eptr_cfg = EPTRKernelConfig(
        eps=eps,
        delta=None,
        x0=x0,
        c_bw=c_bw,
        rf=rf,
        c0=c0,
        null_mode=null_mode,
        seed=seed + 23,
    )
    eptr_out = eptr_nw_gaussian_kernel(x, y, eptr_cfg)

    return [
        {
            "method": "Non-private",
            "theta": float(theta_np),
            "se": pointwise_se(float(theta_np), truth),
            "released": 1.0,
            "truth": truth,
        },
        {
            "method": "WPE",
            "theta": float(wave_out["theta_priv"]),
            "se": pointwise_se(float(wave_out["theta_priv"]), truth),
            "released": 1.0,
            "truth": truth,
        },
        {
            "method": "ePTR",
            "theta": float(eptr_out["theta_priv"]),
            "se": pointwise_se(float(eptr_out["theta_priv"]), truth),
            "released": float(bool(eptr_out["released"])),
            "truth": truth,
        },
    ]


def summarize_rows(rows: list[dict], x_key: str) -> list[dict]:
    bucket = defaultdict(lambda: {"se": [], "released": []})
    for row in rows:
        key = (row[x_key], row["method"])
        bucket[key]["se"].append(row["se"])
        bucket[key]["released"].append(row["released"])

    out = []
    for (xval, method), vals in bucket.items():
        se_arr = np.asarray(vals["se"], dtype=float)
        rel_arr = np.asarray(vals["released"], dtype=float)
        out.append(
            {
                x_key: xval,
                "method": method,
                "mean_se": float(np.mean(se_arr)),
                "sd_se": float(np.std(se_arr, ddof=1)) if len(se_arr) > 1 else 0.0,
                "mean_release": float(np.mean(rel_arr)),
            }
        )
    out.sort(key=lambda d: (float(d[x_key]), METHOD_ORDER.index(d["method"])))
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(summary_rows: list[dict], out_path: Path, n: int, eps: float) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in METHOD_ORDER:
        sub = [r for r in summary_rows if r["method"] == method]
        if not sub:
            continue

        xs = np.array([float(r["a"]) for r in sub], dtype=float)
        ys = np.array([float(r["mean_se"]) for r in sub], dtype=float)
        ss = np.array([float(r["sd_se"]) for r in sub], dtype=float)

        line, = ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.3,
            markersize=6.5,
            label=method,
        )

        color = line.get_color()
        lower = np.maximum(ys - ss, 0.0)
        upper = ys + ss

        ax.fill_between(
            xs,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel(r"Concentration parameter $a$", fontsize=13)
    ax.set_ylabel("Mean pointwise error", fontsize=13)
    ax.set_title(
        rf"Mean pointwise error vs. $a$ ($n={n}$, $\varepsilon={eps}$)",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    ax.set_ylim(bottom=0)

    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=10.5,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-grid", type=str, default="0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,2.9,3.2,3.5")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--c-bw", type=float, default=0.2)
    parser.add_argument("--rf", type=float, default=1.0)
    parser.add_argument("--c0", type=float, default=0.5)
    parser.add_argument("--sigma-eps", type=float, default=0.2)
    parser.add_argument("--null-mode", type=str, default="uniform", choices=["nan", "zero", "uniform"])
    parser.add_argument("--wavelet-L", type=int, default=8)
    parser.add_argument("--wavelet-nu", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="./KernelRegression/results_beta_vs_a")
    args = parser.parse_args()

    a_grid = parse_float_list(args.a_grid)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for i, a in enumerate(a_grid):
        for rep in range(args.reps):
            seed = args.seed + 100000 * i + rep
            res = run_one_rep(
                n=args.n,
                eps=args.eps,
                a=a,
                x0=args.x0,
                c_bw=args.c_bw,
                rf=args.rf,
                c0=args.c0,
                sigma_eps=args.sigma_eps,
                null_mode=args.null_mode,
                wavelet_L=args.wavelet_L,
                wavelet_nu=args.wavelet_nu,
                seed=seed,
            )
            for row in res:
                raw_rows.append(
                    {
                        "a": a,
                        "rep": rep,
                        **row,
                    }
                )

    summary_rows = summarize_rows(raw_rows, "a")

    write_csv(
        outdir / "raw_beta_vs_a.csv",
        raw_rows,
        ["a", "rep", "method", "theta", "se", "released", "truth"],
    )
    write_csv(
        outdir / "summary_beta_vs_a.csv",
        summary_rows,
        ["a", "method", "mean_se", "sd_se", "mean_release"],
    )
    make_plot(summary_rows, outdir / "plot_beta_vs_a.png", n=args.n, eps=args.eps)
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()