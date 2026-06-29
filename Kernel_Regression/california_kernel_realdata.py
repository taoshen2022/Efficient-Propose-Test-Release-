from __future__ import annotations

"""
California Housing real-data experiment for pointwise ePTR kernel regression.

Data:
    sklearn.datasets.fetch_california_housing

Default response:
    Y = MedHouseVal, median house value in units of 100,000 dollars.

Default kernel variables:
    MedInc, Latitude, Longitude, HouseAge

DP-compatible preprocessing:
    - Response Y is scaled using a public range [y_min, y_max]:
          y_scaled = (Y - y_mid) / y_scale,
          y_mid = (y_min + y_max) / 2,
          y_scale = (y_max - y_min) / 2.
      Default: [0, 5], so y_mid = 2.5 and y_scale = 2.5.
    - Covariates X are clipped and scaled using public fixed ranges.
      No train-split mean, sd, median, IQR, min, or max is used.

Default design:
    - 80/20 train/test split in each repetition.
    - The same split and the same held-out query points are used for all eps values
      within one repetition.
    - Privacy budget is parameterized as eps = 2^t.
    - The plot x-axis is log2(eps).

Methods:
    - Public midpoint baseline
    - DP mean baseline
    - Non-private KernelNW
    - ePTR KernelNW
    - WPE, a simple 1D Haar-wavelet private baseline using the first selected feature

Plot:
    Main MSE plots show only non-private, ePTR, and WPE.
    Legend labels are shortened to non-private, ePTR, and WPE.
    ePTR and non-private use the same fixed colors/styles as previous scripts.
    WPE uses a distinct color.
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Public ranges for DP-compatible scaling
# ---------------------------------------------------------------------

# These ranges are fixed in advance and treated as public.
# For the default features, these are broad California Housing dataset ranges.
# If you add new fixed features, also add their public ranges here.
PUBLIC_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "medinc": (0.0, 15.0),
    "houseage": (0.0, 55.0),
    "latitude": (32.0, 42.0),
    "longitude": (-125.0, -114.0),

    # Original sklearn variables, included for optional use.
    "averooms": (0.0, 150.0),
    "avebedrms": (0.0, 35.0),
    "population": (0.0, 40000.0),
    "aveoccup": (0.0, 1300.0),

    # Engineered variables, included for optional use.
    "bedroom_ratio": (0.0, 5.0),
    "medinc_sq": (0.0, 225.0),
    "lat_x_lon": (-5250.0, -3600.0),
    "rooms_x_occup": (0.0, 200000.0),
}


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_csv_list(s: str | None) -> list[str]:
    if s is None or str(s).strip() == "":
        return []
    return [normalize_name(x) for x in str(s).split(",") if str(x).strip()]


def parse_float_list(s: str | None) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def public_mid_scale(lo: float, hi: float) -> tuple[float, float]:
    if hi <= lo:
        raise ValueError(f"Invalid public range [{lo}, {hi}]. Need hi > lo.")
    mid = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    return mid, scale


def scale_response_public(y: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    y_mid, y_scale = public_mid_scale(y_min, y_max)
    y_clip = np.clip(np.asarray(y, dtype=float), y_min, y_max)
    return (y_clip - y_mid) / y_scale


def unscale_response_public(y_scaled: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    y_mid, y_scale = public_mid_scale(y_min, y_max)
    return y_mid + y_scale * np.asarray(y_scaled, dtype=float)


def scale_to_unit_public(
    Xtr_raw: np.ndarray,
    Xte_raw: np.ndarray,
    feature_names: list[str],
    public_ranges: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    Xtr_raw = np.asarray(Xtr_raw, dtype=float)
    Xte_raw = np.asarray(Xte_raw, dtype=float)

    Xtr_scaled = np.zeros_like(Xtr_raw, dtype=float)
    Xte_scaled = np.zeros_like(Xte_raw, dtype=float)

    missing = [c for c in feature_names if normalize_name(c) not in public_ranges]
    if missing:
        raise ValueError(
            "Missing public ranges for features: "
            + ", ".join(missing)
            + "\nAdd them to PUBLIC_FEATURE_RANGES or use the default fixed features."
        )

    for j, c in enumerate(feature_names):
        lo, hi = public_ranges[normalize_name(c)]
        if hi <= lo:
            raise ValueError(f"Invalid public range for {c}: [{lo}, {hi}]")

        Xtr_scaled[:, j] = (np.clip(Xtr_raw[:, j], lo, hi) - lo) / (hi - lo)
        Xte_scaled[:, j] = (np.clip(Xte_raw[:, j], lo, hi) - lo) / (hi - lo)

    return Xtr_scaled, Xte_scaled


# ---------------------------------------------------------------------
# IO and summaries
# ---------------------------------------------------------------------

def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for {path}")

    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)

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
            vals = [float(r[m]) for r in rows if m in r and pd.notna(r[m])]
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


def random_train_test_indices(
    n: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    ntr = int(round(train_frac * n))
    ntr = max(5, min(n - 1, ntr))
    return perm[:ntr], perm[ntr:]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_california_housing_data(
    download_if_missing: bool = True,
    data_home: str | None = None,
    data_path: str | None = None,
    target_column: str = "medhouseval",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load California Housing from sklearn or from a local CSV file.

    For a local CSV, the target column is removed from the feature frame.
    Common target names such as ``MedHouseVal``, ``target``, and
    ``median_house_value`` are also recognized after name normalization.
    """
    if data_path is not None:
        df = pd.read_csv(data_path)
        df.columns = [normalize_name(c) for c in df.columns]

        candidates = [normalize_name(target_column), "medhouseval", "target", "median_house_value"]
        target = next((c for c in candidates if c in df.columns), None)
        if target is None:
            raise ValueError(
                "Cannot find target column in local CSV. Tried: "
                + ", ".join(candidates)
                + f". Available columns: {list(df.columns)}"
            )

        y = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
        Xdf = df.drop(columns=[target]).copy()
        for c in Xdf.columns:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

        keep = np.isfinite(y) & ~Xdf.isna().any(axis=1).to_numpy()
        return Xdf.loc[keep].reset_index(drop=True), y[keep]

    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as e:
        raise ImportError("Please install scikit-learn first: pip install scikit-learn") from e

    data = fetch_california_housing(
        data_home=data_home,
        download_if_missing=download_if_missing,
        as_frame=True,
    )

    Xdf = data.data.copy()
    Xdf.columns = [normalize_name(c) for c in Xdf.columns]

    y = np.asarray(data.target, dtype=float)

    return Xdf, y


def add_california_features(Xdf: pd.DataFrame) -> pd.DataFrame:
    out = Xdf.copy()

    if "averooms" in out.columns and "avebedrms" in out.columns:
        out["bedroom_ratio"] = out["avebedrms"] / np.maximum(out["averooms"], 1e-12)

    if "medinc" in out.columns:
        out["medinc_sq"] = out["medinc"] ** 2

    if "latitude" in out.columns and "longitude" in out.columns:
        out["lat_x_lon"] = out["latitude"] * out["longitude"]

    if "averooms" in out.columns and "aveoccup" in out.columns:
        out["rooms_x_occup"] = out["averooms"] * out["aveoccup"]

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    keep = ~out.isna().any(axis=1)

    return out.loc[keep].reset_index(drop=True)


def resolve_fixed_features(feature_names: list[str], requested: list[str]) -> list[str]:
    if not requested:
        return []

    norm_to_actual = {normalize_name(c): c for c in feature_names}

    out, missing = [], []

    for r in requested:
        nr = normalize_name(r)
        if nr in norm_to_actual:
            out.append(norm_to_actual[nr])
        else:
            missing.append(r)

    if missing:
        raise ValueError(
            "Fixed features not found after preprocessing: "
            + str(missing)
            + "\nAvailable columns: "
            + ", ".join(feature_names)
        )

    seen, deduped = set(), []
    for c in out:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped


def select_features(
    Xdf: pd.DataFrame,
    ytr_scaled: np.ndarray,
    train_idx: np.ndarray,
    fixed_features: list[str],
    max_features: int,
    allow_private_feature_selection: bool,
) -> tuple[list[str], list[dict]]:
    names = list(Xdf.columns)

    if fixed_features:
        selected = resolve_fixed_features(names, fixed_features)
        rows = [
            {
                "feature": c,
                "abs_corr_train": float("nan"),
                "corr_train": float("nan"),
                "selected": 1,
                "selection_mode": "fixed_public",
            }
            for c in selected
        ]
        return selected, rows

    if not allow_private_feature_selection:
        raise ValueError(
            "For DP-compatible preprocessing, --fixed-features must be nonempty. "
            "Correlation-based feature selection is data-dependent. "
            "Use --allow-private-feature-selection only for diagnostic experiments."
        )

    Xtr = Xdf.iloc[train_idx][names].to_numpy(float)

    scores = []

    for j, c in enumerate(names):
        x = Xtr[:, j]
        if np.std(x) <= 1e-12 or np.std(ytr_scaled) <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, ytr_scaled)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0

        scores.append((abs(corr), c, corr))

    scores.sort(reverse=True, key=lambda t: t[0])

    selected = [c for _, c, _ in scores[:max_features]]

    rows = [
        {
            "feature": c,
            "abs_corr_train": float(a),
            "corr_train": float(corr),
            "selected": int(c in selected),
            "selection_mode": "private_correlation_diagnostic",
        }
        for a, c, corr in scores
    ]

    return selected, rows


# ---------------------------------------------------------------------
# Kernel NW and ePTR
# ---------------------------------------------------------------------

def gaussian_kernel_const(d: int) -> float:
    return float((2 * math.pi) ** (-0.5 * d))


def bandwidth_nw(n: int, d: int, c_bw: float) -> float:
    return float(c_bw * n ** (-1.0 / (d + 4.0)))


def gaussian_kernel_weights(Xtr: np.ndarray, Xq: np.ndarray, h: float) -> np.ndarray:
    sq = np.sum((Xq[:, None, :] - Xtr[None, :, :]) ** 2, axis=2)
    d = Xtr.shape[1]
    return gaussian_kernel_const(d) / (h ** d) * np.exp(-0.5 * sq / (h * h))


def nw_predict_many(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xq: np.ndarray,
    h: float,
    rf: Optional[float] = None,
) -> np.ndarray:
    W = gaussian_kernel_weights(Xtr, Xq, h)
    den = np.maximum(W.sum(axis=1), 1e-12)
    out = (W @ ytr) / den
    return np.clip(out, -rf, rf) if rf is not None else out


def eptr_release_probability(gamma: float, eps: float, delta: float) -> tuple[float, float]:
    M = 1.0 + (2.0 / eps) * math.log(max(1.0 / delta, 1.0 / eps))
    logit = 0.5 * eps * (gamma - M)

    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)

    return float(p), float(M)


@dataclass
class EPTRKernelConfig:
    eps: float
    delta: float
    c_bw: float
    rf: float
    c0: float
    null_mode: str
    budget_mode: str
    seed: Optional[int] = None


def eptr_kernel_predict_many(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xq: np.ndarray,
    cfg: EPTRKernelConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    n, d = Xtr.shape
    m = Xq.shape[0]

    eps_q = cfg.eps / m if cfg.budget_mode == "split_total" else cfg.eps
    delta_q = cfg.delta / m if cfg.budget_mode == "split_total" else cfg.delta

    h = bandwidth_nw(n, d, cfg.c_bw)
    ck = gaussian_kernel_const(d)
    yc = np.clip(ytr, -cfg.rf, cfg.rf)

    W = gaussian_kernel_weights(Xtr, Xq, h)
    den = np.maximum(W.sum(axis=1), 1e-12)
    theta = np.clip((W @ yc) / den, -cfg.rf, cfg.rf)

    alpha = 4.0 * cfg.rf * ck / ((h ** d) * cfg.c0 * n)
    noise_sd = (2.0 * alpha / eps_q) * math.sqrt(2.0 * math.log(1.25 / delta_q))

    out = np.zeros(m)
    rel = np.zeros(m, dtype=bool)
    ps = np.zeros(m)
    gam = np.zeros(m)
    deg = np.zeros(m)

    for j in range(m):
        margin = float(den[j] - cfg.c0 * n - 2.0 * ck / (h ** d))
        gamma = max(margin, 0.0) / max(2.0 * ck / (h ** d), 1e-12)

        p_rel, _ = eptr_release_probability(gamma, eps_q, delta_q)
        released = bool(rng.uniform() < p_rel)

        rel[j] = released
        ps[j] = p_rel
        gam[j] = gamma
        deg[j] = float(den[j])

        if released:
            out[j] = float(np.clip(theta[j] + rng.normal(0, noise_sd), -cfg.rf, cfg.rf))
        elif cfg.null_mode == "uniform":
            out[j] = float(rng.uniform(-cfg.rf, cfg.rf))
        else:
            # Deterministic public fallback: midpoint of scaled response range.
            # Since y_scaled is based on [y_min, y_max], this is 0.
            out[j] = 0.0

    return {
        "theta_np": theta,
        "theta_priv": out,
        "release_rate": float(rel.mean()),
        "mean_p_release": float(ps.mean()),
        "min_p_release": float(ps.min()),
        "mean_gamma": float(gam.mean()),
        "min_gamma": float(gam.min()),
        "mean_degree": float(deg.mean()),
        "min_degree": float(deg.min()),
        "noise_sd": float(noise_sd),
        "bandwidth": float(h),
        "alpha": float(alpha),
        "eps_q": float(eps_q),
        "delta_q": float(delta_q),
    }


# ---------------------------------------------------------------------
# Simple DP mean baseline
# ---------------------------------------------------------------------

def dp_mean_predict_many(
    ytr: np.ndarray,
    m: int,
    epsilon: float,
    delta: float,
    rf: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    n = len(ytr)
    yc = np.clip(ytr, -rf, rf)
    theta = float(np.mean(yc))

    sensitivity = 2.0 * rf / max(n, 1)
    noise_sd = (sensitivity / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))

    pred = float(np.clip(theta + rng.normal(0, noise_sd), -rf, rf))

    return {
        "theta_priv": np.full(m, pred, dtype=float),
        "noise_sd": float(noise_sd),
        "sensitivity": float(sensitivity),
    }


# ---------------------------------------------------------------------
# 1D Haar-wavelet private baseline
# ---------------------------------------------------------------------

def haar_father(x: np.ndarray | float) -> np.ndarray | float:
    a = np.asarray(x)
    out = ((a >= 0) & (a < 1)).astype(float)
    return float(out) if np.isscalar(x) else out


def haar_wavelet_lk(
    x: np.ndarray | float,
    level: int,
    k: int,
) -> np.ndarray | float:
    a = np.asarray(x, dtype=float)
    z = (2 ** level) * a - k

    out = np.zeros_like(a, dtype=float)
    amp = 2.0 ** (level / 2.0)

    out[(z >= 0) & (z < 0.5)] = amp
    out[(z >= 0.5) & (z < 1.0)] = -amp

    return float(out) if np.isscalar(x) else out


def choose_wavelet_level(n: int, eps: float, nu: float) -> int:
    D1 = (n * n * eps * eps) ** (1.0 / (2.0 * nu + 2.0))
    D2 = n ** (1.0 / (2.0 * nu + 1.0))
    return max(0, int(math.floor(math.log2(max(1.0, min(D1, D2))))))


def wavelet_point_estimate_haar(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    L: int,
    tau: float,
    rf: Optional[float],
) -> float:
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    y = np.clip(np.asarray(y, dtype=float), -tau, tau)

    theta = float(np.mean(y * haar_father(x))) * float(haar_father(x0))

    if 0 <= x0 < 1:
        for level in range(L + 1):
            k0 = min(int(math.floor((2 ** level) * x0)), 2 ** level - 1)
            theta += (
                float(np.mean(y * haar_wavelet_lk(x, level, k0)))
                * float(haar_wavelet_lk(x0, level, k0))
            )

    return float(np.clip(theta, -rf, rf)) if rf is not None else float(theta)


def wavelet_nrdp_predict_many(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xq: np.ndarray,
    epsilon: float,
    rf: float,
    tau: Optional[float],
    L: Optional[int],
    nu: float,
    rng: np.random.Generator,
    feature_index: int,
) -> dict[str, object]:
    if not (0 <= feature_index < Xtr.shape[1]):
        raise ValueError("wavelet-feature-index outside selected feature range")

    n = Xtr.shape[0]

    x = Xtr[:, feature_index]
    xq = Xq[:, feature_index]

    tau_used = rf if tau is None else float(tau)
    L_used = choose_wavelet_level(n, epsilon, nu) if L is None else int(L)

    theta_hat = np.asarray([
        wavelet_point_estimate_haar(x, ytr, float(x0), L_used, tau_used, rf)
        for x0 in xq
    ])

    sensitivity = 6.0 * tau_used * (2 ** L_used) / max(n, 1)
    lap_scale = sensitivity / epsilon

    theta_priv = np.clip(theta_hat + rng.laplace(0, lap_scale, size=len(xq)), -rf, rf)

    return {
        "theta_priv": theta_priv,
        "lap_scale": float(lap_scale),
        "wavelet_L": int(L_used),
        "sensitivity": float(sensitivity),
    }


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_one_rep(
    Xdf: pd.DataFrame,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    query_pos: np.ndarray,
    args: argparse.Namespace,
    rep_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    tr, te = train_idx, test_idx

    ytr_raw = y[tr]
    yte_raw = y[te]

    y_mid, y_scale = public_mid_scale(args.y_min, args.y_max)

    ytr = scale_response_public(ytr_raw, args.y_min, args.y_max)

    selected_names, feature_rows = select_features(
        Xdf,
        ytr,
        tr,
        parse_csv_list(args.fixed_features),
        args.max_kernel_features,
        args.allow_private_feature_selection,
    )

    X_all = Xdf[selected_names].to_numpy(float)

    Xtr_raw, Xte_raw = X_all[tr], X_all[te]
    Xtr, Xte = scale_to_unit_public(
        Xtr_raw,
        Xte_raw,
        selected_names,
        PUBLIC_FEATURE_RANGES,
    )

    q = query_pos[query_pos < len(te)]

    Xq = Xte[q]
    yq_raw = yte_raw[q]
    yq = scale_response_public(yq_raw, args.y_min, args.y_max)

    records: list[dict] = []

    selected_rows = [
        {
            "epsilon": epsilon,
            "t_log2_epsilon": math.log2(epsilon),
            "rep": rep_seed,
            "selected_feature": c,
            "n_features": len(selected_names),
        }
        for c in selected_names
    ]

    # Public deterministic midpoint baseline.
    pred_mid_scaled = np.zeros_like(yq)
    pred_mid_original = np.full_like(yq_raw, y_mid)

    records.append({
        "method": "Public midpoint baseline",
        "epsilon": epsilon,
        "t_log2_epsilon": math.log2(epsilon),
        "rep": rep_seed,
        "mse_test": mse(yq, pred_mid_scaled),
        "mse_test_original_scale": mse(yq_raw, pred_mid_original),
        "release_rate": 1.0,
        "n_train": len(tr),
        "n_test": len(q),
        "n_features": len(selected_names),
    })

    out_mean = dp_mean_predict_many(
        ytr,
        len(q),
        epsilon,
        delta,
        args.kernel_rf,
        np.random.default_rng(rep_seed + 11),
    )
    pred_dm = np.asarray(out_mean["theta_priv"], dtype=float)

    records.append({
        "method": "DP mean baseline",
        "epsilon": epsilon,
        "t_log2_epsilon": math.log2(epsilon),
        "rep": rep_seed,
        "mse_test": mse(yq, pred_dm),
        "mse_test_original_scale": mse(
            yq_raw,
            unscale_response_public(pred_dm, args.y_min, args.y_max),
        ),
        "release_rate": 1.0,
        "noise_sd": out_mean["noise_sd"],
        "sensitivity": out_mean["sensitivity"],
        "n_train": len(tr),
        "n_test": len(q),
        "n_features": len(selected_names),
    })

    h = bandwidth_nw(Xtr.shape[0], Xtr.shape[1], args.kernel_c_bw)
    pred_np = nw_predict_many(Xtr, ytr, Xq, h, rf=args.kernel_rf)

    records.append({
        "method": "Non-private KernelNW",
        "epsilon": epsilon,
        "t_log2_epsilon": math.log2(epsilon),
        "rep": rep_seed,
        "mse_test": mse(yq, pred_np),
        "mse_test_original_scale": mse(
            yq_raw,
            unscale_response_public(pred_np, args.y_min, args.y_max),
        ),
        "release_rate": 1.0,
        "bandwidth": float(h),
        "n_train": len(tr),
        "n_test": len(q),
        "n_features": len(selected_names),
    })

    out_k = eptr_kernel_predict_many(
        Xtr,
        ytr,
        Xq,
        EPTRKernelConfig(
            eps=epsilon,
            delta=delta,
            c_bw=args.kernel_c_bw,
            rf=args.kernel_rf,
            c0=args.kernel_c0,
            null_mode=args.kernel_null_mode,
            budget_mode=args.kernel_budget_mode,
            seed=rep_seed + 101,
        ),
    )
    pred_eptr = np.asarray(out_k["theta_priv"], dtype=float)

    records.append({
        "method": "ePTR KernelNW",
        "epsilon": epsilon,
        "t_log2_epsilon": math.log2(epsilon),
        "rep": rep_seed,
        "mse_test": mse(yq, pred_eptr),
        "mse_test_original_scale": mse(
            yq_raw,
            unscale_response_public(pred_eptr, args.y_min, args.y_max),
        ),
        "release_rate": out_k["release_rate"],
        "mean_p_release": out_k["mean_p_release"],
        "min_p_release": out_k["min_p_release"],
        "mean_gamma": out_k["mean_gamma"],
        "min_gamma": out_k["min_gamma"],
        "mean_degree": out_k["mean_degree"],
        "min_degree": out_k["min_degree"],
        "noise_sd": out_k["noise_sd"],
        "bandwidth": out_k["bandwidth"],
        "alpha": out_k["alpha"],
        "eps_q": out_k["eps_q"],
        "delta_q": out_k["delta_q"],
        "n_train": len(tr),
        "n_test": len(q),
        "n_features": len(selected_names),
    })

    if args.include_wavelet:
        wave_out = wavelet_nrdp_predict_many(
            Xtr,
            ytr,
            Xq,
            epsilon,
            args.wavelet_rf if args.wavelet_rf is not None else args.kernel_rf,
            args.wavelet_tau,
            args.wavelet_L,
            args.wavelet_nu,
            np.random.default_rng(rep_seed + 202),
            args.wavelet_feature_index,
        )

        pred_w = np.asarray(wave_out["theta_priv"], dtype=float)

        records.append({
            "method": "WPE",
            "epsilon": epsilon,
            "t_log2_epsilon": math.log2(epsilon),
            "rep": rep_seed,
            "mse_test": mse(yq, pred_w),
            "mse_test_original_scale": mse(
                yq_raw,
                unscale_response_public(pred_w, args.y_min, args.y_max),
            ),
            "release_rate": 1.0,
            "lap_scale": wave_out["lap_scale"],
            "wavelet_L": wave_out["wavelet_L"],
            "sensitivity": wave_out["sensitivity"],
            "wavelet_feature": (
                selected_names[args.wavelet_feature_index]
                if args.wavelet_feature_index < len(selected_names)
                else ""
            ),
            "n_train": len(tr),
            "n_test": len(q),
            "n_features": len(selected_names),
        })

    for row in feature_rows:
        row.update({
            "epsilon": epsilon,
            "t_log2_epsilon": math.log2(epsilon),
            "rep": rep_seed,
        })

    return records, feature_rows, selected_rows


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

METHOD_LABEL = {
    "Non-private KernelNW": "non-private",
    "ePTR KernelNW": "ePTR",
    "WPE": "WPE",
    "Public midpoint baseline": "Public midpoint",
    "DP mean baseline": "DP mean",
}

METHOD_STYLE = {
    "Non-private KernelNW": {
        "color": "#66A61E",
        "linestyle": ":",
        "marker": "s",
        "open": False,
    },
    "ePTR KernelNW": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "open": True,
    },
    "WPE": {
        "color": "#7E57C2",
        "linestyle": "--",
        "marker": "D",
        "open": True,
    },
    "Public midpoint baseline": {
        "color": "#999999",
        "linestyle": "-.",
        "marker": "x",
        "open": False,
    },
    "DP mean baseline": {
        "color": "#E69F00",
        "linestyle": "--",
        "marker": "^",
        "open": False,
    },
}


def _set_log2_eps_axis_ticks(ax, eps_values: np.ndarray) -> None:
    eps_values = np.asarray(eps_values, dtype=float)
    eps_values = eps_values[np.isfinite(eps_values) & (eps_values > 0)]

    if len(eps_values) == 0:
        return

    xvals = np.log2(np.sort(np.unique(eps_values)))
    ax.set_xticks(xvals)
    ax.set_xticklabels([f"{x:g}" for x in xvals])


def plot_metric(
    summary: list[dict],
    out_path: Path,
    metric: str,
    ylabel: str,
    title: str,
    methods: list[str],
    errorbar: str = "sd",
) -> None:
    df = pd.DataFrame(summary)

    ycol = f"mean_{metric}"

    if errorbar == "se":
        errcol = f"se_{metric}"
    elif errorbar == "sd":
        errcol = f"sd_{metric}"
    else:
        errcol = ""

    if ycol not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in methods:
        g = df[df["method"] == method].sort_values("epsilon")

        if g.empty:
            continue

        eps_arr = g["epsilon"].to_numpy(float)
        xs = np.log2(eps_arr)
        ys = g[ycol].to_numpy(float)

        yerr = g[errcol].to_numpy(float) if errcol and errcol in g.columns else None

        style = METHOD_STYLE.get(method, {})

        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")
        markerfacecolor = "white" if style.get("open", False) else color

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            label=METHOD_LABEL.get(method, method),
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
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)

    ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        fontsize=10.5,
        loc="best",
    )

    ax.set_ylim(bottom=0)

    _set_log2_eps_axis_ticks(ax, df["epsilon"].dropna().unique().astype(float))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="California Housing ePTR KernelNW experiment")

    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--train-frac", type=float, default=0.2)

    p.add_argument(
        "--eps-powers",
        type=str,
        default="0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25",
        help="Comma-separated t values. The script uses epsilon = 2^t.",
    )

    p.add_argument("--delta", type=float, default=1e-2)
    p.add_argument("--kernel-eval-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="california_kernel_eptr_outputs")
    p.add_argument("--data-home", type=str, default=None)
    p.add_argument("--data-path", type=str, default=None, help="Optional local California Housing CSV file.")
    p.add_argument("--target-column", type=str, default="MedHouseVal", help="Target column name for --data-path.")
    p.add_argument("--no-download", dest="download_if_missing", action="store_false")

    p.add_argument(
        "--fixed-features",
        type=str,
        default="medinc,latitude,longitude,houseage",
        help="Use public fixed features. For DP-compatible preprocessing, keep this nonempty.",
    )

    p.add_argument(
        "--allow-private-feature-selection",
        action="store_true",
        help="Allow correlation feature selection using the private train split. Diagnostic only, not DP-compatible.",
    )

    p.add_argument(
        "--max-kernel-features",
        type=int,
        default=4,
        help="Used only when fixed-features is empty and --allow-private-feature-selection is set.",
    )

    p.add_argument(
        "--y-min",
        type=float,
        default=0.0,
        help="Public lower bound for response. Default 0 for MedHouseVal.",
    )
    p.add_argument(
        "--y-max",
        type=float,
        default=5.0,
        help="Public upper bound for response. Default 5 for MedHouseVal.",
    )

    p.add_argument("--kernel-c0", type=float, default=0.3)
    p.add_argument("--kernel-c-bw", type=float, default=0.5)
    p.add_argument("--kernel-rf", type=float, default=1.0)
    p.add_argument("--kernel-null-mode", type=str, default="zero", choices=["zero", "uniform"])
    p.add_argument(
        "--kernel-budget-mode",
        type=str,
        default="full_per_query",
        choices=["full_per_query", "split_total"],
    )

    p.add_argument("--include-wavelet", action="store_true", default=True)
    p.add_argument("--no-wavelet", dest="include_wavelet", action="store_false")
    p.add_argument("--wavelet-L", type=int, default=10)
    p.add_argument("--wavelet-nu", type=float, default=1.0)
    p.add_argument("--wavelet-tau", type=float, default=None)
    p.add_argument("--wavelet-rf", type=float, default=1.0)
    p.add_argument("--wavelet-feature-index", type=int, default=0)

    p.add_argument(
        "--errorbar",
        type=str,
        default="sd",
        choices=["sd", "se", "none"],
        help="Use SD/SE/none as error bars. No shaded bands are used.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    eps_powers = parse_float_list(args.eps_powers)
    if not eps_powers:
        raise ValueError("--eps-powers cannot be empty.")

    eps_values = [2.0 ** t for t in eps_powers]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Xdf, y = load_california_housing_data(
        download_if_missing=args.download_if_missing,
        data_home=args.data_home,
        data_path=args.data_path,
        target_column=args.target_column,
    )

    Xdf = add_california_features(Xdf)

    # add_california_features only removes rows with missing features;
    # California housing has no missing rows.
    y = y[:len(Xdf)]

    save_csv(
        outdir / "available_features_after_preprocessing.csv",
        [{"feature": c} for c in Xdf.columns],
    )

    save_csv(
        outdir / "public_feature_ranges.csv",
        [
            {"feature": k, "public_min": v[0], "public_max": v[1]}
            for k, v in sorted(PUBLIC_FEATURE_RANGES.items())
        ],
    )

    split_info = []

    for rep in range(args.reps):
        split_seed = args.seed + 10000 * rep
        rng = np.random.default_rng(split_seed)

        tr, te = random_train_test_indices(len(y), args.train_frac, rng)

        m_eval = min(args.kernel_eval_size, len(te))
        q = rng.choice(len(te), size=m_eval, replace=False)

        split_info.append((tr, te, q, split_seed))

    raw, feature_rows, selected_rows = [], [], []

    for rep, (tr, te, q, split_seed) in enumerate(split_info):
        for t, eps in zip(eps_powers, eps_values):
            rep_seed = split_seed + int(round(1000 * eps))

            rec, fs, sel = run_one_rep(
                Xdf,
                y,
                eps,
                args.delta,
                tr,
                te,
                q,
                args,
                rep_seed,
            )

            raw.extend(rec)
            feature_rows.extend(fs)
            selected_rows.extend(sel)

            print(f"done rep={rep + 1}/{args.reps}, t={t:g}, eps={eps:g}")

    metrics = [
        "mse_test",
        "mse_test_original_scale",
        "release_rate",
        "mean_p_release",
        "min_p_release",
        "mean_gamma",
        "min_gamma",
        "mean_degree",
        "min_degree",
        "noise_sd",
        "bandwidth",
        "alpha",
        "eps_q",
        "delta_q",
        "lap_scale",
        "wavelet_L",
        "sensitivity",
        "n_features",
    ]

    summary = summarize_records(raw, ["method", "epsilon"], metrics)

    y_mid, y_scale = public_mid_scale(args.y_min, args.y_max)

    meta = [{
        "n_total_rows_used": int(len(y)),
        "response_label": "MedHouseVal in units of 100000 dollars",
        "response_scaling": "public range scaling: y_scaled=(y-y_mid)/y_scale",
        "y_min_public": args.y_min,
        "y_max_public": args.y_max,
        "y_mid_public": y_mid,
        "y_scale_public": y_scale,
        "x_scaling": "public feature ranges; clipped to range then mapped to [0,1]",
        "train_frac": args.train_frac,
        "reps": args.reps,
        "kernel_eval_size": args.kernel_eval_size,
        "same_split_and_query_across_eps": True,
        "fixed_features": args.fixed_features,
        "allow_private_feature_selection": args.allow_private_feature_selection,
        "kernel_c_bw": args.kernel_c_bw,
        "kernel_c0": args.kernel_c0,
        "kernel_rf": args.kernel_rf,
        "kernel_budget_mode": args.kernel_budget_mode,
        "delta": args.delta,
        "seed": args.seed,
        "errorbar": args.errorbar,
        "eps_powers": " ".join(map(str, eps_powers)),
        "eps_values": " ".join(map(str, eps_values)),
        "x_axis": "log2(epsilon)",
        "plot_methods": "Non-private KernelNW|ePTR KernelNW|WPE",
        "legend_labels": "non-private|ePTR|WPE",
    }]

    save_csv(outdir / "raw_results.csv", raw)
    save_csv(outdir / "summary_results.csv", summary)
    save_csv(outdir / "feature_selection_by_rep.csv", feature_rows)
    save_csv(outdir / "selected_features_by_rep.csv", selected_rows)
    save_csv(outdir / "run_metadata.csv", meta)

    main_methods = ["Non-private KernelNW", "ePTR KernelNW", "WPE"]

    plot_metric(
        summary,
        outdir / "mse_vs_eps_standardized_response.png",
        "mse_test",
        "Test MSE on public-scaled response",
        "MSE vs privacy budget on California housing data",
        main_methods,
        errorbar=args.errorbar,
    )

    plot_metric(
        summary,
        outdir / "mse_vs_eps_original_scale.png",
        "mse_test_original_scale",
        "Test MSE on response scale",
        "California Housing: MSE vs privacy budget",
        main_methods,
        errorbar=args.errorbar,
    )

    plot_metric(
        summary,
        outdir / "release_rate_vs_eps.png",
        "release_rate",
        "Release rate",
        "ePTR release rate vs privacy budget",
        ["ePTR KernelNW"],
        errorbar=args.errorbar,
    )

    print("\nSaved:")
    for name in [
        "raw_results.csv",
        "summary_results.csv",
        "mse_vs_eps_standardized_response.png",
        "mse_vs_eps_original_scale.png",
        "release_rate_vs_eps.png",
        "available_features_after_preprocessing.csv",
        "public_feature_ranges.csv",
        "run_metadata.csv",
    ]:
        print(outdir / name)

    print("\nSummary preview:")
    for row in summary[:15]:
        print(row)


if __name__ == "__main__":
    main()