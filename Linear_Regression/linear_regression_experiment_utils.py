from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from linear_data import LinearSimConfig, generate_linear_regression_data
from eptr_linear_regression import EPTRLinearConfig, run_single_demo as run_eptr_demo
from dp_ols_sheffet_alg1 import DPOLSAlg1Config, run_single_demo as run_sheffet_demo
from dp_linear_regression_cai2021 import DPCaiLRConfig, run_single_demo as run_cai_demo
from dp_functional_mechanism_lr import DPFMLinearConfig, run_single_demo as run_fm_demo


@dataclass
class ExperimentDefaults:
    p: int = 20
    tau: float = 0.2
    sigma_eps: float = 1.0
    n_test: int = 100_000
    reps: int = 30

    eptr_c0: float = 0.17
    eptr_Rx: float = 3.0
    eptr_Rtheta: float = 1.0

    sheffet_Rx: float = 3.0
    sheffet_Ry: float = 9.0

    cai_Rx: float = 3.0
    cai_R: float = 4.0
    cai_C: float = 1.0

    fm_Rx: float = 3.0
    fm_Ry: float = 9.0
    fm_C: float | None = None
    fm_min_eig_floor: float = 1e-6
    fm_jitter: float = 1e-8
    fm_lambda_reg: float | None = None
    fm_use_spectral_trimming: bool = True




def exact_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)



def prediction_mse(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    resid = y_test - X_test @ beta
    return float(np.mean(resid ** 2))



def signal_mse(beta: np.ndarray, X_eval: np.ndarray, theta_true: np.ndarray) -> float:
    diff = X_eval @ (beta - theta_true)
    return float(np.mean(diff ** 2))



def parameter_sq_error(beta: np.ndarray, beta_true: np.ndarray) -> float:
    return float(np.sum((beta - beta_true) ** 2))





def run_all_methods_one_rep(
    *,
    n: int,
    epsilon: float,
    defaults: ExperimentDefaults,
    rep_seed: int,
) -> list[dict[str, float | int | str | bool]]:
    sim_cfg = LinearSimConfig(
        n=n,
        p=defaults.p,
        tau=defaults.tau,
        sigma_eps=defaults.sigma_eps,
        n_test=defaults.n_test,
        seed=rep_seed,
    )
    data = generate_linear_regression_data(sim_cfg)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    beta_np = exact_ols(data["X_train"], data["y_train"])

    records: list[dict[str, float | int | str | bool]] = []
    delta = 0.01

    records.append(
        {
            "method": "Non-private",
            "n": n,
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": prediction_mse(beta_np, data["X_test"], data["y_test"]),
            "signal_mse": signal_mse(beta_np, data["X_test"], theta_true),
            "param_err": parameter_sq_error(beta_np, theta_true),
            "released": True,
        }
    )

    eptr_cfg = EPTRLinearConfig(
        epsilon=epsilon,
        delta=delta,
        c0=defaults.eptr_c0,
        Rx=defaults.eptr_Rx,
        Rtheta=defaults.eptr_Rtheta,
        null_output="zero",
        seed=10_000 + rep_seed,
    )
    out_eptr = run_eptr_demo(sim_cfg, eptr_cfg)
    theta_eptr = np.asarray(out_eptr["theta_priv"], dtype=float)
    records.append(
        {
            "method": "ePTR",
            "n": n,
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": float(out_eptr["mse_test_private"]),
            "signal_mse": signal_mse(theta_eptr, data["X_test"], theta_true),
            "param_err": float(out_eptr.get("param_err_private", parameter_sq_error(theta_eptr, theta_true))),
            "released": bool(out_eptr["released"]),
        }
    )

    sheffet_cfg = DPOLSAlg1Config(
        epsilon=epsilon,
        delta=delta,
        r=None,
        Rx=defaults.sheffet_Rx,
        Ry=defaults.sheffet_Ry,
        seed=20_000 + rep_seed,
    )
    out_sheffet = run_sheffet_demo(sim_cfg, sheffet_cfg)
    beta_sheffet = np.asarray(out_sheffet["beta_private"], dtype=float)
    records.append(
        {
            "method": "DPJL",
            "n": n,
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": float(out_sheffet["mse_test_private"]),
            "signal_mse": signal_mse(beta_sheffet, data["X_test"], theta_true),
            "param_err": float(out_sheffet.get("param_err_private", parameter_sq_error(beta_sheffet, theta_true))),
            "released": not bool(out_sheffet["altered"]),
        }
    )


    cai_cfg = DPCaiLRConfig(
        epsilon=epsilon,
        delta=delta,
        T=30,
        eta0=0.8,
        Rx=4.5,
        R=defaults.cai_R,
        C=defaults.cai_C,
        seed=30_000 + rep_seed,
    )
    out_cai = run_cai_demo(sim_cfg, cai_cfg)
    beta_cai = np.asarray(out_cai["beta_priv"], dtype=float)
    records.append(
        {
            "method": "DPGD",
            "n": n,
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": float(out_cai["mse_test_private"]),
            "signal_mse": signal_mse(beta_cai, data["X_test"], theta_true),
            "param_err": float(out_cai.get("param_err_private", parameter_sq_error(beta_cai, theta_true))),
            "released": True,
        }
    )

    fm_cfg = DPFMLinearConfig(
        epsilon=epsilon,
        Rx=defaults.fm_Rx,
        Ry=defaults.fm_Ry,
        C=defaults.fm_C,
        min_eig_floor=defaults.fm_min_eig_floor,
        jitter=defaults.fm_jitter,
        lambda_reg=defaults.fm_lambda_reg,
        use_spectral_trimming=defaults.fm_use_spectral_trimming,
        seed=35_000 + rep_seed,
    )
    out_fm = run_fm_demo(sim_cfg, fm_cfg)
    beta_fm = np.asarray(out_fm["beta_private"], dtype=float)
    records.append(
        {
            "method": "FM",
            "n": n,
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": float(out_fm["mse_test_private"]),
            "signal_mse": signal_mse(beta_fm, data["X_test"], theta_true),
            "param_err": float(out_fm.get("param_err_private", parameter_sq_error(beta_fm, theta_true))),
            "released": True,
            "convexified": bool(out_fm.get("convexified", False)),
            "spectral_trimmed": bool(out_fm.get("spectral_trimmed", False)),
            "lambda_reg": float(out_fm.get("lambda_reg", 0.0)),
        }
    )



    return records



def summarize_records(
    records: Sequence[dict[str, float | int | str | bool]],
    *,
    x_key: str,
    metric_key: str,
) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[str, float], list[float]] = {}
    for row in records:
        x_val = float(row[x_key])
        method = str(row["method"])
        metric_val = float(row[metric_key])
        buckets.setdefault((method, x_val), []).append(metric_val)

    summary: list[dict[str, float | int | str]] = []
    mean_name = f"mean_{metric_key}"
    sd_name = f"sd_{metric_key}"
    se_name = f"se_{metric_key}"
    for (method, x_val), values in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        se = float(sd / np.sqrt(arr.size)) if arr.size > 0 else np.nan
        summary.append(
            {
                "method": method,
                x_key: x_val,
                mean_name: mean,
                sd_name: sd,
                se_name: se,
                "n_rep": int(arr.size),
            }
        )
    return summary



def save_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_summary(
    summary_rows: Sequence[dict[str, object]],
    *,
    x_key: str,
    x_label: str,
    y_label: str,
    title: str,
    output_prefix: str | Path,
    use_log_x: bool = False,
) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        raise ValueError("summary_rows is empty.")

    preferred_order = ["Non-private", "ePTR", "DPJL", "DPGD", "FM"]

    methods_present = []
    for row in summary_rows:
        m = str(row["method"])
        if m not in methods_present:
            methods_present.append(m)

    methods = [m for m in preferred_order if m in methods_present]
    methods += [m for m in methods_present if m not in methods]

    mean_key_candidates = [k for k in summary_rows[0].keys() if k.startswith("mean_")]
    sd_key_candidates = [k for k in summary_rows[0].keys() if k.startswith("sd_")]
    if len(mean_key_candidates) != 1 or len(sd_key_candidates) != 1:
        raise ValueError("summary_rows must contain exactly one mean_* and one sd_* metric column")

    mean_key = mean_key_candidates[0]
    sd_key = sd_key_candidates[0]

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in methods:
        rows = [r for r in summary_rows if str(r["method"]) == method]
        xs = np.array([float(r[x_key]) for r in rows], dtype=float)
        ys = np.array([float(r[mean_key]) for r in rows], dtype=float)
        ysd = np.array([float(r[sd_key]) for r in rows], dtype=float)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        ysd = ysd[order]

        line, = ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.3,
            markersize=6.5,
            label=method,
        )

        color = line.get_color()
        lower = np.maximum(ys - ysd, 0.0)
        upper = ys + ysd

        ax.fill_between(
            xs,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    if use_log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
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
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)