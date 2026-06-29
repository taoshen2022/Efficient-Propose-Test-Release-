import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional

# ============================================================
# Self-contained Bayes simulation utilities
# ============================================================


def clip_to_ball(X: np.ndarray, Rx: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if Rx is None or np.isinf(Rx):
        return X.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.minimum(1.0, Rx / np.maximum(norms, 1e-12))
    return X * scale



def build_class_means(K: int, p: int, delta_sep: float) -> np.ndarray:
    means = np.zeros((K, p), dtype=float)
    for k in range(K):
        means[k, k % p] = delta_sep
    return means



def generate_bayes_data(
    n: int,
    K: int,
    mu: np.ndarray,
    delta_sep: float,
    p: int,
    Rx: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Generate Gaussian naive-Bayes data with identity covariance.
    Class k has mean vector m_k and covariance I_p.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    means = build_class_means(K=K, p=p, delta_sep=delta_sep)

    y = rng.choice(K, size=n, p=mu)
    X = means[y] + rng.normal(loc=0.0, scale=1.0, size=(n, p))
    if Rx is not None:
        X = clip_to_ball(X, Rx)
    return X, y, mu, means



def fit_empirical_bayes(X: np.ndarray, y: np.ndarray, K: Optional[int] = None):
    if K is None:
        K = int(np.max(y)) + 1
    n, p = X.shape
    counts = np.bincount(y, minlength=K)
    mu_hat = counts / n

    means_hat = np.zeros((K, p), dtype=float)
    for k in range(K):
        if counts[k] > 0:
            means_hat[k] = X[y == k].mean(axis=0)

    return mu_hat, means_hat, counts



def fit_bayes_direct_wi13(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if Rx <= 0:
        raise ValueError("Rx must be positive.")
    if not (0 < eps_split_counts < 1):
        raise ValueError("eps_split_counts must lie in (0,1).")

    rng = np.random.default_rng(seed)

    n, p = X.shape
    if K is None:
        K = int(np.max(y)) + 1

    X_clip = clip_to_ball(X, Rx)
    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_lap_scale = 2.0 / eps_count
    counts_tilde = counts_hat.astype(float) + rng.laplace(loc=0.0, scale=count_lap_scale, size=K)
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)
    means_tilde = means_hat.copy()

    for k in range(K):
        nk = counts_hat[k]
        sens_mean_k = 2.0 * Rx / (nk + 1.0)
        scale_mean_k = sens_mean_k / eps_per_mean_coord
        means_tilde[k, :] = means_hat[k, :] + rng.laplace(loc=0.0, scale=scale_mean_k, size=p)

    means_tilde = clip_to_ball(means_tilde, Rx)
    return {"mu_tilde": mu_tilde, "means_tilde": means_tilde}



def sample_cauchy(rng: np.random.Generator, loc: float = 0.0, scale: float = 1.0, size=None):
    return loc + scale * rng.standard_cauchy(size=size)



def mean_sensitivity_proxy(n_k: int, Rx: float) -> float:
    return 2.0 * Rx / (n_k + 1.0)



def fit_bayes_direct_cauchy(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    Rx: float,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    eps_split_counts: float = 0.2,
):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if Rx <= 0:
        raise ValueError("Rx must be positive.")
    if not (0 < eps_split_counts < 1):
        raise ValueError("eps_split_counts must lie in (0,1).")

    rng = np.random.default_rng(seed)

    n, p = X.shape
    if K is None:
        K = int(np.max(y)) + 1

    X_clip = clip_to_ball(X, Rx)
    mu_hat, means_hat, counts_hat = fit_empirical_bayes(X_clip, y, K=K)

    eps_count = eps_split_counts * eps
    eps_mean_total = (1.0 - eps_split_counts) * eps

    count_cauchy_scale = np.sqrt(2.0) * 2.0 / eps_count
    counts_tilde = counts_hat.astype(float) + sample_cauchy(
        rng=rng, loc=0.0, scale=count_cauchy_scale, size=K
    )
    counts_tilde = np.maximum(counts_tilde, 1e-8)
    mu_tilde = counts_tilde / counts_tilde.sum()

    eps_per_mean_coord = eps_mean_total / (K * p)
    means_tilde = means_hat.copy()

    for k in range(K):
        nk = counts_hat[k]
        s_k = mean_sensitivity_proxy(nk, Rx)
        scale_mean_k = np.sqrt(2.0) * s_k / eps_per_mean_coord
        means_tilde[k, :] = means_hat[k, :] + sample_cauchy(
            rng=rng, loc=0.0, scale=scale_mean_k, size=p
        )

    means_tilde = clip_to_ball(means_tilde, Rx)
    return {"mu_tilde": mu_tilde, "means_tilde": means_tilde}



def sigmoid(z: float) -> float:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))



def random_fallback_parameter(K: int, p: int, c0: float, Rx: float, rng: np.random.Generator):
    mu_rand = rng.dirichlet(np.ones(K))
    mu_rand = np.maximum(mu_rand, c0)
    mu_rand = mu_rand / mu_rand.sum()

    means_rand = rng.normal(loc=0.0, scale=Rx, size=(K, p))
    means_rand = clip_to_ball(means_rand, Rx)

    theta_rand = np.concatenate([mu_rand, means_rand.ravel()])
    return theta_rand



def etpr_release_vector(
    theta_hat: np.ndarray,
    gamma: float,
    alpha: float,
    eps: float,
    delta: float,
    K: int,
    p: int,
    c0: float,
    Rx: float,
    rng: np.random.Generator,
):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1).")

    M = 1.0 + (2.0 / eps) * np.log(max(1.0 / delta, 1.0 / eps))
    p_release = sigmoid(0.5 * eps * (gamma - M))
    passed = rng.uniform() < p_release

    noise_sd = (2.0 * alpha / eps) * np.sqrt(2.0 * np.log(1.25 / delta))

    if passed:
        theta_tilde = theta_hat + rng.normal(loc=0.0, scale=noise_sd, size=theta_hat.shape)
    else:
        theta_tilde = random_fallback_parameter(K=K, p=p, c0=c0, Rx=Rx, rng=rng)

    return theta_tilde



def fit_bayes_etpr(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    delta: float,
    Rx: float,
    c0: float,
    K: int = 3,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)

    n, p = X.shape
    X_clip = clip_to_ball(X, Rx)

    mu_hat, means_hat, counts = fit_empirical_bayes(X_clip, y, K=K)

    gamma = max(float(counts.min() - c0 * n - 1.0), 0.0)
    alpha = (2.0 / n) * np.sqrt(2.0 * Rx**2 / c0**2 + 2.0)

    theta_hat = np.concatenate([mu_hat, means_hat.ravel()])
    theta_tilde = etpr_release_vector(
        theta_hat=theta_hat,
        gamma=gamma,
        alpha=alpha,
        eps=eps,
        delta=delta,
        K=K,
        p=p,
        c0=c0,
        Rx=Rx,
        rng=rng,
    )

    mu_tilde = theta_tilde[:K].copy()
    means_tilde = theta_tilde[K:].reshape(K, p).copy()

    mu_tilde = np.maximum(mu_tilde, c0)
    mu_tilde = mu_tilde / mu_tilde.sum()
    means_tilde = clip_to_ball(means_tilde, Rx)

    return {"mu_tilde": mu_tilde, "means_tilde": means_tilde}



def predict_naive_bayes(X: np.ndarray, mu: np.ndarray, means: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    means = np.asarray(means, dtype=float)

    if np.any(mu <= 0):
        raise ValueError("All entries of mu must be positive.")

    scores = X @ means.T - 0.5 * np.sum(means**2, axis=1) + np.log(mu)
    return np.argmax(scores, axis=1)



def misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))



def balanced_error_rate(y_true: np.ndarray, y_pred: np.ndarray, K: Optional[int] = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if K is None:
        K = int(max(np.max(y_true), np.max(y_pred))) + 1

    class_errors = []
    for k in range(K):
        idx = (y_true == k)
        if np.any(idx):
            class_errors.append(np.mean(y_pred[idx] != y_true[idx]))
    if not class_errors:
        return 0.0
    return float(np.mean(class_errors))



def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, K: int) -> float:
    if metric == "misclassification":
        return misclassification_rate(y_true, y_pred)
    if metric == "balanced_error":
        return balanced_error_rate(y_true, y_pred, K=K)
    raise ValueError(f"Unknown metric: {metric}")



def evaluate_one_rep(
    n_train,
    n_test,
    eps,
    delta,
    mu_true,
    delta_sep,
    p,
    Rx,
    c0,
    K,
    eps_split_counts,
    rep_seed,
    metric,
):
    train_seed = 1000 + rep_seed
    test_seed = 900000 + rep_seed

    X_train, y_train, _, _ = generate_bayes_data(
        n=n_train,
        K=K,
        mu=mu_true,
        delta_sep=delta_sep,
        p=p,
        Rx=None,
        seed=train_seed,
    )
    X_test, y_test, _, _ = generate_bayes_data(
        n=n_test,
        K=K,
        mu=mu_true,
        delta_sep=delta_sep,
        p=p,
        Rx=None,
        seed=test_seed,
    )

    out = {}

    X_train_clip = clip_to_ball(X_train, Rx)
    mu_hat, means_hat, _ = fit_empirical_bayes(X_train_clip, y_train, K=K)
    y_pred_np = predict_naive_bayes(X_test, mu_hat, means_hat)
    out["Nonprivate"] = compute_metric(y_test, y_pred_np, metric=metric, K=K)

    etpr = fit_bayes_etpr(
        X=X_train, y=y_train, eps=eps, delta=delta, Rx=Rx, c0=c0, K=K, seed=3000 + rep_seed
    )
    out["ePTR"] = compute_metric(
        y_test,
        predict_naive_bayes(X_test, etpr["mu_tilde"], etpr["means_tilde"]),
        metric=metric,
        K=K,
    )

    wi13 = fit_bayes_direct_wi13(
        X=X_train,
        y=y_train,
        eps=eps,
        Rx=Rx,
        K=K,
        seed=4000 + rep_seed,
        eps_split_counts=eps_split_counts,
    )
    out["dpnb"] = compute_metric(
        y_test,
        predict_naive_bayes(X_test, wi13["mu_tilde"], wi13["means_tilde"]),
        metric=metric,
        K=K,
    )

    cauchy = fit_bayes_direct_cauchy(
        X=X_train,
        y=y_train,
        eps=eps,
        Rx=Rx,
        K=K,
        seed=5000 + rep_seed,
        eps_split_counts=eps_split_counts,
    )
    out["dpnbss"] = compute_metric(
        y_test,
        predict_naive_bayes(X_test, cauchy["mu_tilde"], cauchy["means_tilde"]),
        metric=metric,
        K=K,
    )

    return out



def summarize_results(df, x_col):
    mean_std = (
        df.groupby([x_col, "method"])["error"]
        .agg(["mean", "std"])
        .reset_index()
    )

    quants = (
        df.groupby([x_col, "method"])["error"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
    )
    quants.columns = [x_col, "method", "q25", "q75"]

    grouped = mean_std.merge(quants, on=[x_col, "method"], how="left")
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped = grouped.sort_values([x_col, "method"]).reset_index(drop=True)
    return grouped



def plot_summary(summary, x_col, x_label, title, out_png, metric):
    label_map = {
        "Nonprivate": "Non-private",
        "ePTR": "ePTR",
        "dpnb": "DPNB",
        "dpnbss": "DPNBSS",
    }
    method_order = ["Nonprivate", "ePTR", "dpnb", "dpnbss"]

    ylabel = (
        "Balanced test error"
        if metric == "balanced_error"
        else "Test misclassification rate"
    )

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for method in method_order:
        sub = summary.loc[summary["method"] == method].sort_values(x_col)
        if sub.empty:
            continue

        x = sub[x_col].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()

        line, = ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.3,
            markersize=6.5,
            label=label_map.get(method, method),
        )

        color = line.get_color()

        ax.fill_between(
            x,
            y - s,
            y + s,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=10.5,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Plot test error versus pi_min for Bayes privacy baselines.")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=100000)
    parser.add_argument("--outdir", type=str, default="./Classification/results_pimin")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--delta-sep", type=float, default=3.0)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--Rx", type=float, default=8.0)
    parser.add_argument("--eps-split-counts", type=float, default=0.2)
    parser.add_argument(
        "--metric",
        type=str,
        default="balanced_error",
        choices=["misclassification", "balanced_error"],
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pimin_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    K = 3

    rows = []
    for pi_min in pimin_grid:
        mu_true = np.array([0.7 - pi_min, 0.3, pi_min], dtype=float)
        c0_current = 0.5 * pi_min

        for rep in range(args.reps):
            errs = evaluate_one_rep(
                n_train=args.n_train,
                n_test=args.n_test,
                eps=args.eps,
                delta=args.delta,
                mu_true=mu_true,
                delta_sep=args.delta_sep,
                p=args.p,
                Rx=args.Rx,
                c0=c0_current,
                K=K,
                eps_split_counts=args.eps_split_counts,
                rep_seed=100000 * int(round(100 * pi_min)) + rep,
                metric=args.metric,
            )
            for method, err in errs.items():
                rows.append({
                    "pi_min": pi_min,
                    "c0": c0_current,
                    "rep": rep,
                    "method": method,
                    "error": err,
                })

    raw = pd.DataFrame(rows)
    summary = summarize_results(raw, "pi_min")

    raw_csv = outdir / f"bayes_error_vs_pimin_{args.metric}_raw.csv"
    summary_csv = outdir / f"bayes_error_vs_pimin_{args.metric}_summary.csv"
    plot_png = outdir / f"bayes_error_vs_pimin_{args.metric}.png"

    raw.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    metric_title = (
        "Balanced test error"
        if args.metric == "balanced_error"
        else "Test misclassification rate"
    )

    plot_summary(
        summary=summary,
        x_col="pi_min",
        x_label=r"Minimum class probability $\pi_{\min}$",
        title=rf"{metric_title} vs. $\pi_{{\min}}$ ($n={args.n_train}$, $\varepsilon={args.eps}$)",
        out_png=plot_png,
        metric=args.metric,
    )

    print(f"Saved raw results to: {raw_csv}")
    print(f"Saved summary results to: {summary_csv}")
    print(f"Saved plot to: {plot_png}")


if __name__ == "__main__":
    main()
