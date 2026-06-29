from __future__ import annotations

"""
Wine Quality real-data experiment: linear-regression-only version.

Methods:
    - non-private
    - ePTR
    - DPJL
    - DPGD
    - FM

Main design:
    - One train/test split is generated per replication.
    - The same split is reused for all epsilon values within that replication.
    - X is standardized using training data only.
    - y is scaled using public constants y_mid and y_scale:
          y_scaled = (y - y_mid) / y_scale.
    - Test MSE is reported on the original wine-quality scale.
    - ePTR diagnostics are saved separately.
    - If ePTR does not release OLS, it releases a deterministic public fallback
      value on the public-scaled response scale.

Plot style:
    - Error bars instead of shaded bands.
    - Fixed colors for ePTR and non-private.
    - Distinct colors for DPJL, DPGD, and FM.
    - Privacy budget is parameterized as eps = 2^t.
    - The x-axis is log2(eps).
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TARGET = "quality"

DEFAULT_LINEAR_FEATURES = [
    "alcohol",
    "volatile acidity",
    "density",
    "chlorides",
    "free sulfur dioxide",
    "residual sugar",
    "pH",
]


# ---------------------------------------------------------------------
# IO and summaries
# ---------------------------------------------------------------------

def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save: {path}")

    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize_records(
    records: list[dict],
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}

    for r in records:
        buckets.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    out: list[dict] = []

    for key, rows in sorted(buckets.items(), key=lambda kv: kv[0]):
        res = {k: v for k, v in zip(group_keys, key)}
        res["n_rep"] = len(rows)

        for m in metric_keys:
            vals = []
            for r in rows:
                if m in r and r[m] is not None:
                    try:
                        v = float(r[m])
                        if np.isfinite(v):
                            vals.append(v)
                    except Exception:
                        pass

            a = np.asarray(vals, dtype=float)

            res[f"mean_{m}"] = float(np.mean(a)) if len(a) else float("nan")
            res[f"sd_{m}"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            res[f"se_{m}"] = (
                float(res[f"sd_{m}"] / math.sqrt(len(a))) if len(a) else float("nan")
            )

        out.append(res)

    return out


def mse(y: np.ndarray, yh: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yh)) ** 2))


def scale_response_public(y: np.ndarray, y_mid: float, y_scale: float) -> np.ndarray:
    if y_scale <= 0:
        raise ValueError("y_scale must be positive.")
    return (np.asarray(y, dtype=float) - y_mid) / y_scale


def unscale_response_public(y_scaled: np.ndarray, y_mid: float, y_scale: float) -> np.ndarray:
    if y_scale <= 0:
        raise ValueError("y_scale must be positive.")
    return y_mid + y_scale * np.asarray(y_scaled, dtype=float)


# ---------------------------------------------------------------------
# Wine data
# ---------------------------------------------------------------------

def read_wine_csv(path: str | Path, wine_type: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    if wine_type is not None:
        df["wine_type"] = wine_type

    return df


def load_wine_data(
    data_path: Optional[str],
    red_path: Optional[str],
    white_path: Optional[str],
) -> pd.DataFrame:
    if data_path:
        return read_wine_csv(data_path)

    parts = []

    if red_path:
        parts.append(read_wine_csv(red_path, "red"))

    if white_path:
        parts.append(read_wine_csv(white_path, "white"))

    if not parts:
        raise ValueError("Provide --data-path, or --red-path/--white-path.")

    return pd.concat(parts, axis=0, ignore_index=True)


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_wine_dataframe(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    if target_name not in df.columns:
        raise ValueError(f"Target {target_name!r} not found. Columns={list(df.columns)}")

    out = df.copy()

    for c in out.columns:
        if c != "wine_type":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(axis=0, how="any").reset_index(drop=True)

    if out.empty:
        raise ValueError("No complete rows after cleaning.")

    return out


def prepare_design(
    df: pd.DataFrame,
    target_name: str,
    feature_names: list[str],
    add_wine_type: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = pd.to_numeric(df[target_name], errors="coerce").astype(float)

    feats = [c for c in feature_names if c in df.columns and c != target_name]

    if add_wine_type and "wine_type" in df.columns and "wine_type" not in feats:
        feats.append("wine_type")

    if not feats:
        raise ValueError(
            f"No requested features found. Requested={feature_names}; "
            f"columns={list(df.columns)}"
        )

    Xdf = df[feats].copy()
    cat = [c for c in Xdf.columns if Xdf[c].dtype == object or c == "wine_type"]

    for c in Xdf.columns:
        if c not in cat:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    if cat:
        Xdf = pd.get_dummies(Xdf, columns=cat, drop_first=True, dtype=float)

    keep = ~(y.isna() | Xdf.isna().any(axis=1))

    X = Xdf.loc[keep].to_numpy(float)
    yy = y.loc[keep].to_numpy(float)
    names = list(Xdf.columns)

    if X.shape[0] == 0:
        raise ValueError("No valid rows after feature processing.")

    return X, yy, names


# ---------------------------------------------------------------------
# Linear regression and DP helpers
# ---------------------------------------------------------------------

def random_train_test_indices(
    n: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    ntr = int(round(train_frac * n))
    ntr = max(5, min(n - 1, ntr))
    return perm[:ntr], perm[ntr:]


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def clip_rows_by_l2(X: np.ndarray, radius: float) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X * np.minimum(1.0, radius / np.maximum(norm, 1e-12))


def clip_vector(y: np.ndarray, radius: float) -> np.ndarray:
    return np.clip(y, -radius, radius)


def project_l2_ball(beta: np.ndarray, radius: Optional[float]) -> np.ndarray:
    if radius is None:
        return beta

    nrm = float(np.linalg.norm(beta))

    if nrm <= radius or nrm == 0.0:
        return beta

    return beta * (radius / nrm)


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(p), X.T @ y)


def ols_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X) @ y


def eptr_release_probability(gamma: float, eps: float, delta: float) -> tuple[float, float]:
    if eps <= 0:
        raise ValueError("epsilon must be positive")

    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    M = 1.0 + (2.0 / eps) * math.log(max(1.0 / delta, 1.0 / eps))
    logit = 0.5 * eps * (gamma - M)

    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)

    return float(p), float(M)


@dataclass
class EPTRLinearConfig:
    epsilon: float = 1.0
    delta: float = 1e-2
    c0: float = 0.03
    rx: float = 1.5
    ry: float = 3.0
    rtheta: float = 4.0
    ridge: float = 1e-8
    fallback_value_scaled: float = 0.0
    seed: Optional[int] = None


def eptr_linear_regression(
    Xtr: np.ndarray,
    ytr_scaled: np.ndarray,
    Xte: np.ndarray,
    cfg: EPTRLinearConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    Xc = clip_rows_by_l2(Xtr, cfg.rx)
    yc = clip_vector(ytr_scaled, cfg.ry)

    n, p = Xc.shape

    XtX = Xc.T @ Xc
    lam_min = float(np.linalg.eigvalsh(XtX).min())

    good_margin = lam_min - n * cfg.c0 - 2.0 * cfg.rx**2
    gamma = max(good_margin, 0.0) / max(2.0 * cfg.rx**2, 1e-12)

    p_release, M = eptr_release_probability(gamma, cfg.epsilon, cfg.delta)
    released = bool(rng.uniform() < p_release)

    beta = project_l2_ball(fit_ridge(Xc, yc, cfg.ridge), cfg.rtheta)

    sens_l2 = 4.0 * cfg.rx * cfg.ry / (n * max(cfg.c0, 1e-12))
    noise_sd = (
        sens_l2 / cfg.epsilon
    ) * math.sqrt(2.0 * math.log(1.25 / cfg.delta))

    if released:
        beta_priv = project_l2_ball(
            beta + rng.normal(0.0, noise_sd, size=p),
            cfg.rtheta,
        )
        y_pred_scaled = Xte @ beta_priv
        fallback_to_deterministic = False
        deterministic_fallback_scaled = float("nan")
    else:
        deterministic_fallback_scaled = float(
            np.clip(cfg.fallback_value_scaled, -cfg.ry, cfg.ry)
        )

        beta_priv = np.zeros(p)
        y_pred_scaled = np.full(Xte.shape[0], deterministic_fallback_scaled, dtype=float)
        fallback_to_deterministic = True

    return {
        "y_pred_scaled": y_pred_scaled,
        "released": released,
        "fallback_to_deterministic": fallback_to_deterministic,
        "deterministic_fallback_scaled": deterministic_fallback_scaled,
        "p_release": p_release,
        "gamma": gamma,
        "M": M,
        "lam_min": lam_min,
        "lam_min_over_n": lam_min / n,
        "good_margin_raw": good_margin,
        "threshold_n_c0": n * cfg.c0,
        "stability_penalty_2rx2": 2.0 * cfg.rx**2,
        "noise_sd": noise_sd,
        "sens_l2": sens_l2,
        "p": p,
        "rank": int(np.linalg.matrix_rank(Xc)),
    }


# ---------------------------------------------------------------------
# Competitor methods
# ---------------------------------------------------------------------

def dp_sheffet_alg1_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    Rx: float,
    Ry: float,
    r: Optional[int],
    rng: np.random.Generator,
) -> dict[str, object]:
    Xc = clip_rows_by_l2(X, Rx)
    yc = clip_vector(y, Ry)

    A = np.column_stack([Xc, yc])
    n, d = A.shape

    rr = int(r) if r is not None else int(max(d + 10, min(n, 5 * d)))
    rr = max(rr, d + 1)

    B = float(math.sqrt(Rx**2 + Ry**2))
    log_term = math.log(8.0 / delta)

    w_sq = (8.0 * B**2 / epsilon) * (
        math.sqrt(2.0 * rr * log_term) + 2.0 * log_term
    )
    w = float(math.sqrt(max(w_sq, 1e-12)))

    Z = float(rng.laplace(0.0, 4.0 * B**2 / epsilon))

    sigma_min_sq = float(np.linalg.eigvalsh(A.T @ A).min())
    threshold = float(
        w_sq + Z + (4.0 * B**2 * math.log(1.0 / delta) / epsilon)
    )

    if sigma_min_sq > threshold:
        Aproj = rng.normal(0.0, 1.0, size=(rr, n)) @ A
        altered = False
    else:
        Aprime = np.vstack([A, w * np.eye(d)])
        Aproj = rng.normal(0.0, 1.0, size=(rr, n + d)) @ Aprime
        altered = True

    return {
        "beta": ols_pinv(Aproj[:, :-1], Aproj[:, -1]),
        "altered": altered,
        "sigma_min_sq": sigma_min_sq,
        "threshold": threshold,
        "w": w,
        "r": rr,
    }


def default_step_size(X: np.ndarray) -> float:
    gram = (X.T @ X) / max(X.shape[0], 1)
    return 1.0 / max(float(np.linalg.eigvalsh(gram).max()), 1e-8)


def dp_cai2021_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    Rx: float,
    R: float,
    C: float,
    T: Optional[int],
    eta0: Optional[float],
    rng: np.random.Generator,
) -> dict[str, object]:
    n, p = X.shape

    Xc = clip_rows_by_l2(X, Rx)
    yc = clip_vector(y, R)

    eta = float(eta0) if eta0 is not None else default_step_size(Xc)
    TT = int(T) if T is not None else int(math.ceil(5.0 * math.log(max(n, 2))))

    B = 4.0 * (R + C * Rx) * Rx
    noise_sd = (
        eta * B * math.sqrt(2.0 * math.log(2.0 * TT / delta))
    ) / (n * (epsilon / TT))

    beta = np.zeros(p)

    for _ in range(TT):
        grad = (Xc.T @ (Xc @ beta - yc)) / n
        beta = project_l2_ball(
            beta - eta * grad + rng.normal(0.0, noise_sd, size=p),
            C,
        )

    return {
        "beta": beta,
        "T": TT,
        "eta0": eta,
        "B": B,
        "noise_sd": noise_sd,
    }


def dp_functional_mechanism_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    Rx: float,
    Ry: float,
    C: Optional[float],
    lambda_reg: Optional[float],
    use_spectral_trimming: bool,
    rng: np.random.Generator,
) -> dict[str, object]:
    Xc = clip_rows_by_l2(X, Rx)
    yc = clip_vector(y, Ry)

    Xu = Xc / Rx
    yu = yc / Ry

    _, p = Xu.shape

    M = Xu.T @ Xu
    b = Xu.T @ yu
    linear = -2.0 * b

    sens = 2.0 * (p + 1) ** 2
    scale = sens / epsilon

    linear_noisy = linear + rng.laplace(0.0, scale, size=p)

    upper_noise = rng.laplace(0.0, scale, size=(p, p))
    M_noisy = np.zeros_like(M)

    for j in range(p):
        for k in range(j, p):
            M_noisy[j, k] = M[j, k] + upper_noise[j, k]
            M_noisy[k, j] = M_noisy[j, k]

    M_sym = 0.5 * (M_noisy + M_noisy.T)

    lam = (
        float(lambda_reg)
        if lambda_reg is not None
        else float(4.0 * math.sqrt(2.0) * scale)
    )

    eigvals, eigvecs = np.linalg.eigh(M_sym + lam * np.eye(p))

    if use_spectral_trimming:
        mask = eigvals > 0.0
        if not np.any(mask):
            mask = np.ones_like(eigvals, dtype=bool)
            eigvals = np.maximum(eigvals, 1e-10)

        V, L = eigvecs[:, mask], eigvals[mask]
    else:
        V, L = eigvecs, np.maximum(eigvals, 1e-10)

    beta = (Ry / Rx) * (V @ (-0.5 * (V.T @ linear_noisy) / L))

    return {
        "beta": project_l2_ball(beta, C),
        "laplace_scale": scale,
        "lambda_reg": lam,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

METHOD_ORDER = [
    "non-private",
    "ePTR",
    "DPJL",
    "DPGD",
    "FM",
]

METHOD_STYLE = {
    "non-private": {
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
    "DPJL": {
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "D",
        "open": True,
    },
    "DPGD": {
        "color": "#CC79A7",
        "linestyle": "-.",
        "marker": "^",
        "open": True,
    },
    "FM": {
        "color": "#E69F00",
        "linestyle": "-",
        "marker": "v",
        "open": False,
    },
}


def _set_power_axis_ticks(ax, eps_values: np.ndarray) -> None:
    eps_values = np.asarray(eps_values, dtype=float)
    eps_values = eps_values[np.isfinite(eps_values) & (eps_values > 0)]

    if len(eps_values) == 0:
        return

    xs = np.log2(np.sort(np.unique(eps_values)))

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])


def plot_summary(
    summary: list[dict],
    outdir: Path,
    *,
    shade: str = "se",
    skip_dpgd: bool = False,
    title: str = "MSE vs privacy budget on wine quality data",
    filename: str = "mse_vs_eps.png",
) -> None:
    outdir = Path(outdir)
    df = pd.DataFrame(summary).copy()

    if skip_dpgd:
        df = df[df["method"] != "DPGD"]

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in METHOD_ORDER:
        g = df[df["method"] == method].sort_values("epsilon")

        if g.empty:
            continue

        eps_arr = g["epsilon"].to_numpy(float)
        xs = np.log2(eps_arr)
        ys = g["mean_mse_test"].to_numpy(float)

        if shade != "none":
            band_col = f"{shade}_mse_test"
            yerr = g[band_col].to_numpy(float) if band_col in g.columns else None
        else:
            yerr = None

        style = METHOD_STYLE.get(method, {})

        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")
        markerfacecolor = "white" if style.get("open", False) else color

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            label=method,
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
    ax.set_ylabel("Test MSE", fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)

    ax.set_ylim(bottom=0.4)

    _set_power_axis_ticks(ax, df["epsilon"].dropna().unique().astype(float))

    ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        fontsize=10.5,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_release_rate(summary: list[dict], outdir: Path) -> None:
    df = pd.DataFrame(summary)
    g = df[df["method"] == "ePTR"].sort_values("epsilon")

    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    eps_arr = g["epsilon"].to_numpy(float)
    xs = np.log2(eps_arr)
    ys = g["mean_release_rate"].to_numpy(float)

    style = METHOD_STYLE["ePTR"]

    ax.errorbar(
        xs,
        ys,
        yerr=g["se_release_rate"].to_numpy(float) if "se_release_rate" in g.columns else None,
        marker=style["marker"],
        color=style["color"],
        linestyle=style["linestyle"],
        markerfacecolor="white",
        markeredgecolor=style["color"],
        markeredgewidth=1.8,
        linewidth=2.4,
        markersize=7,
        capsize=4,
        capthick=1.6,
        elinewidth=1.6,
        label="ePTR",
    )

    ax.set_xlabel(r"$\log_2(\varepsilon)$", fontsize=13)
    ax.set_ylabel("Release rate", fontsize=13)
    ax.set_title("ePTR release rate vs privacy budget", fontsize=16, fontweight="bold", pad=10)

    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, which="major", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)

    _set_power_axis_ticks(ax, g["epsilon"].dropna().unique().astype(float))

    ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        fontsize=10.5,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(outdir / "release_rate_vs_eps.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_one_rep(
    Xlin: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lin_cfg: EPTRLinearConfig,
    rep_seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    tr, te = train_idx, test_idx

    Xl_tr, Xl_te = Xlin[tr], Xlin[te]
    ytr, yte = y[tr], y[te]

    # X standardization still uses training data.
    # For a fully strict DP preprocessing pipeline, replace this by public/fixed
    # clipping and scaling for X as well.
    xmu, xsd = fit_standardizer(Xl_tr)
    Xl_trs = apply_standardizer(Xl_tr, xmu, xsd)
    Xl_tes = apply_standardizer(Xl_te, xmu, xsd)

    # Public response scaling: no private training response mean/sd.
    ytrs = scale_response_public(ytr, args.y_mid, args.y_scale)

    records: list[dict] = []

    # Non-private OLS/ridge.
    beta_np = fit_ridge(Xl_trs, ytrs, ridge=1e-8)
    yhat_np_scaled = Xl_tes @ beta_np
    yhat_np = unscale_response_public(yhat_np_scaled, args.y_mid, args.y_scale)

    records.append({
        "method": "non-private",
        "epsilon": epsilon,
        "rep": rep_seed,
        "mse_test": mse(yte, yhat_np),
        "release_rate": 1.0,
        "n_train": len(tr),
        "n_test": len(te),
        "n_features": Xl_trs.shape[1],
    })

    # ePTR.
    out_lin = eptr_linear_regression(Xl_trs, ytrs, Xl_tes, lin_cfg)
    yhat_eptr = unscale_response_public(
        np.asarray(out_lin["y_pred_scaled"]),
        args.y_mid,
        args.y_scale,
    )

    records.append({
        "method": "ePTR",
        "epsilon": epsilon,
        "rep": rep_seed,
        "mse_test": mse(yte, yhat_eptr),
        "release_rate": 1.0 if out_lin["released"] else 0.0,
        "fallback_to_deterministic": 1.0 if out_lin["fallback_to_deterministic"] else 0.0,
        "n_train": len(tr),
        "n_test": len(te),
        "n_features": Xl_trs.shape[1],
    })

    diag = {
        "epsilon": epsilon,
        "rep": rep_seed,
        "n_train": len(tr),
    }

    diag.update({
        k: out_lin[k]
        for k in [
            "p_release",
            "gamma",
            "M",
            "lam_min",
            "lam_min_over_n",
            "good_margin_raw",
            "threshold_n_c0",
            "stability_penalty_2rx2",
            "noise_sd",
            "sens_l2",
            "fallback_to_deterministic",
            "deterministic_fallback_scaled",
            "p",
            "rank",
        ]
    })

    # DPJL.
    if args.include_sheffet:
        out = dp_sheffet_alg1_linear(
            Xl_trs,
            ytrs,
            epsilon=epsilon,
            delta=delta,
            Rx=args.sheffet_Rx,
            Ry=args.sheffet_Ry,
            r=args.sheffet_r,
            rng=np.random.default_rng(rep_seed + 301),
        )

        pred_scaled = Xl_tes @ np.asarray(out["beta"])
        pred = unscale_response_public(pred_scaled, args.y_mid, args.y_scale)

        records.append({
            "method": "DPJL",
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": mse(yte, pred),
            "release_rate": 1.0,
            "n_train": len(tr),
            "n_test": len(te),
            "n_features": Xl_trs.shape[1],
            "altered_rate": 1.0 if out["altered"] else 0.0,
        })

    # DPGD.
    if args.include_cai:
        out = dp_cai2021_linear(
            Xl_trs,
            ytrs,
            epsilon=epsilon,
            delta=delta,
            Rx=args.cai_Rx,
            R=args.cai_R,
            C=args.cai_C,
            T=args.cai_T,
            eta0=args.cai_eta0,
            rng=np.random.default_rng(rep_seed + 401),
        )

        pred_scaled = Xl_tes @ np.asarray(out["beta"])
        pred = unscale_response_public(pred_scaled, args.y_mid, args.y_scale)

        records.append({
            "method": "DPGD",
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": mse(yte, pred),
            "release_rate": 1.0,
            "n_train": len(tr),
            "n_test": len(te),
            "n_features": Xl_trs.shape[1],
            "noise_sd": float(out["noise_sd"]),
            "T": int(out["T"]),
        })

    # FM.
    if args.include_fm:
        out = dp_functional_mechanism_linear(
            Xl_trs,
            ytrs,
            epsilon=epsilon,
            Rx=args.fm_Rx,
            Ry=args.fm_Ry,
            C=args.fm_C,
            lambda_reg=args.fm_lambda_reg,
            use_spectral_trimming=not args.fm_no_spectral_trimming,
            rng=np.random.default_rng(rep_seed + 501),
        )

        pred_scaled = Xl_tes @ np.asarray(out["beta"])
        pred = unscale_response_public(pred_scaled, args.y_mid, args.y_scale)

        records.append({
            "method": "FM",
            "epsilon": epsilon,
            "rep": rep_seed,
            "mse_test": mse(yte, pred),
            "release_rate": 1.0,
            "n_train": len(tr),
            "n_test": len(te),
            "n_features": Xl_trs.shape[1],
            "laplace_scale": float(out["laplace_scale"]),
        })

    return records, [diag]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wine Quality linear-only DP regression experiment."
    )

    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--red-path", type=str, default="./wine+quality/winequality-red.csv")
    p.add_argument("--white-path", type=str, default="./wine+quality/winequality-white.csv")
    p.add_argument("--target-name", type=str, default=DEFAULT_TARGET)

    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--train-frac", type=float, default=0.8)

    p.add_argument(
        "--eps-powers",
        type=float,
        nargs="+",
        default=[
            0.0, 0.25, 0.5, 0.75, 1.0,
            1.25, 1.5, 1.75, 2.0, 2.25,
            2.5, 2.75, 3.0, 3.25,
        ],
        help="Use eps = 2^t for each supplied t. The x-axis is log2(eps).",
    )

    p.add_argument("--delta", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="wine_quality_linear_eptr_outputs")

    p.add_argument(
        "--linear-feature-names",
        type=str,
        default="alcohol,volatile acidity,density,chlorides,free sulfur dioxide,residual sugar,pH",
    )
    p.add_argument("--add-wine-type", action="store_true")

    p.add_argument(
        "--y-mid",
        type=float,
        default=5.0,
        help="Public response midpoint. For wine quality, default 5 corresponds to range [0,10].",
    )
    p.add_argument(
        "--y-scale",
        type=float,
        default=5.0,
        help="Public response scale. For wine quality, default 5 maps range [0,10] to [-1,1].",
    )

    p.add_argument("--linear-c0", type=float, default=0.062)
    p.add_argument("--linear-rx", type=float, default=2.0)
    p.add_argument("--linear-ry", type=float, default=3.0)
    p.add_argument("--linear-rtheta", type=float, default=4.0)

    p.add_argument(
        "--linear-fallback-scaled",
        type=float,
        default=0.0,
        help="Deterministic fallback value on the public-scaled response scale. Default 0 maps to y_mid.",
    )

    p.add_argument("--include-sheffet", "--include-dpjl", dest="include_sheffet", action="store_true", default=True)
    p.add_argument("--include-cai", "--include-dpgd", dest="include_cai", action="store_true", default=True)
    p.add_argument("--include-fm", dest="include_fm", action="store_true", default=True)

    p.add_argument("--no-sheffet", "--no-dpjl", dest="include_sheffet", action="store_false")
    p.add_argument("--no-cai", "--no-dpgd", dest="include_cai", action="store_false")
    p.add_argument("--no-fm", dest="include_fm", action="store_false")

    p.add_argument("--sheffet-Rx", type=float, default=10.0)
    p.add_argument("--sheffet-Ry", type=float, default=10.0)
    p.add_argument("--sheffet-r", type=int, default=50)

    p.add_argument("--cai-Rx", type=float, default=1.0)
    p.add_argument("--cai-R", type=float, default=3.0)
    p.add_argument("--cai-C", type=float, default=4.0)
    p.add_argument("--cai-T", type=int, default=None)
    p.add_argument("--cai-eta0", type=float, default=None)

    p.add_argument("--fm-Rx", type=float, default=2.0)
    p.add_argument("--fm-Ry", type=float, default=3.0)
    p.add_argument("--fm-C", type=float, default=4.0)
    p.add_argument("--fm-lambda-reg", type=float, default=None)
    p.add_argument("--fm-no-spectral-trimming", action="store_true")

    p.add_argument(
        "--shade",
        type=str,
        default="se",
        choices=["se", "sd", "none"],
        help="Use SE/SD/none as error bars. No shaded bands are used.",
    )

    p.add_argument("--skip-cai-in-plot", "--skip-dpgd-in-plot", dest="skip_dpgd_in_plot", action="store_true")
    p.add_argument("--plot-title", type=str, default="MSE vs privacy budget on wine quality data")

    return p


def main() -> None:
    args = build_parser().parse_args()

    eps_values = [2.0 ** t for t in args.eps_powers]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = clean_wine_dataframe(
        load_wine_data(args.data_path, args.red_path, args.white_path),
        args.target_name,
    )

    Xlin, y, lin_names = prepare_design(
        df,
        args.target_name,
        parse_csv_list(args.linear_feature_names),
        args.add_wine_type,
    )

    split_info = []
    for rep in range(args.reps):
        split_seed = args.seed + 10_000 * rep
        rng_split = np.random.default_rng(split_seed)
        tr, te = random_train_test_indices(len(y), args.train_frac, rng_split)
        split_info.append((tr, te, split_seed))

    raw: list[dict] = []
    diag: list[dict] = []

    for rep, (tr, te, split_seed) in enumerate(split_info):
        for eps in eps_values:
            rep_seed = split_seed + int(round(1000 * eps))

            lin_cfg = EPTRLinearConfig(
                epsilon=eps,
                delta=args.delta,
                c0=args.linear_c0,
                rx=args.linear_rx,
                ry=args.linear_ry,
                rtheta=args.linear_rtheta,
                fallback_value_scaled=args.linear_fallback_scaled,
                seed=rep_seed + 1,
            )

            r, d = run_one_rep(
                Xlin,
                y,
                epsilon=eps,
                delta=args.delta,
                train_idx=tr,
                test_idx=te,
                lin_cfg=lin_cfg,
                rep_seed=rep_seed,
                args=args,
            )

            raw.extend(r)
            diag.extend(d)

            print(f"done rep={rep + 1}/{args.reps}, eps={eps:g}")

    summary = summarize_records(
        raw,
        ["method", "epsilon"],
        [
            "mse_test",
            "release_rate",
            "fallback_to_deterministic",
            "altered_rate",
            "noise_sd",
            "laplace_scale",
            "n_features",
        ],
    )

    diag_summary = summarize_records(
        diag,
        ["epsilon"],
        [
            "p_release",
            "gamma",
            "M",
            "lam_min",
            "lam_min_over_n",
            "good_margin_raw",
            "threshold_n_c0",
            "stability_penalty_2rx2",
            "noise_sd",
            "sens_l2",
            "fallback_to_deterministic",
            "deterministic_fallback_scaled",
            "p",
            "rank",
        ],
    )

    meta = [{
        "n_total_rows_used": int(len(y)),
        "n_linear_features": int(Xlin.shape[1]),
        "linear_feature_names": "|".join(lin_names),
        "target_name": args.target_name,
        "reps": args.reps,
        "train_frac": args.train_frac,
        "same_split_across_eps": True,
        "eps_powers": " ".join(map(str, args.eps_powers)),
        "eps_values": " ".join(map(str, eps_values)),
        "x_axis": "log2(epsilon)",
        "delta": args.delta,
        "seed": args.seed,
        "response_scaling": "public affine scaling: y_scaled=(y-y_mid)/y_scale",
        "y_mid": args.y_mid,
        "y_scale": args.y_scale,
        "linear_c0": args.linear_c0,
        "linear_rx": args.linear_rx,
        "linear_ry": args.linear_ry,
        "linear_rtheta": args.linear_rtheta,
        "linear_fallback_scaled": args.linear_fallback_scaled,
        "deterministic_fallback_original_scale": (
            args.y_mid + args.y_scale * args.linear_fallback_scaled
        ),
        "fallback_interpretation": "deterministic public fallback on scaled response; default maps to y_mid",
        "include_dpjl": args.include_sheffet,
        "include_dpgd": args.include_cai,
        "include_fm": args.include_fm,
        "errorbar": args.shade,
        "skip_dpgd_in_plot": args.skip_dpgd_in_plot,
    }]

    save_csv(outdir / "raw_results.csv", raw)
    save_csv(outdir / "summary_results.csv", summary)
    save_csv(outdir / "linear_eptr_diagnostics_summary.csv", diag_summary)
    save_csv(outdir / "run_metadata.csv", meta)

    plot_summary(
        summary,
        outdir,
        shade=args.shade,
        skip_dpgd=args.skip_dpgd_in_plot,
        title=args.plot_title,
        filename="mse_vs_eps.png",
    )

    plot_summary(
        summary,
        outdir,
        shade=args.shade,
        skip_dpgd=True,
        title=args.plot_title,
        filename="mse_vs_eps_no_dpgd.png",
    )

    plot_release_rate(summary, outdir)

    print("\nSaved:")
    print(outdir / "raw_results.csv")
    print(outdir / "summary_results.csv")
    print(outdir / "linear_eptr_diagnostics_summary.csv")
    print(outdir / "run_metadata.csv")
    print(outdir / "mse_vs_eps.png")
    print(outdir / "mse_vs_eps_no_dpgd.png")
    print(outdir / "release_rate_vs_eps.png")

    print("\nSummary preview:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()