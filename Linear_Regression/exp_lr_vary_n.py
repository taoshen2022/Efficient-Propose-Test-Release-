from __future__ import annotations

import argparse
from pathlib import Path

from linear_regression_experiment_utils import (
    ExperimentDefaults,
    plot_summary,
    run_all_methods_one_rep,
    save_csv,
    summarize_records,
)

METRICS = [
    ("mse_test", "Test MSE", "mse"),
    ("signal_mse", r"Mean $(X\theta - X\hat\theta)^2$", "signal_mse"),
    ("param_err", r"Parameter error $\|\hat\theta-\theta\|_2^2$", "param_err"),
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Linear-regression experiment: vary n with epsilon fixed, compare all methods on test MSE, signal MSE, and parameter error."
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    )
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--sigma_eps", type=float, default=1.0)
    parser.add_argument("--n_test", type=int, default=100000)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=20260410)
    parser.add_argument("--outdir", type=str, default="results_vary_n")

    parser.add_argument("--eptr_c0", type=float, default=0.5)
    parser.add_argument("--eptr_Rx", type=float, default=4.0)
    parser.add_argument("--eptr_Rtheta", type=float, default=1.0)

    parser.add_argument("--sheffet_Rx", type=float, default=4.0)
    parser.add_argument("--sheffet_Ry", type=float, default=4.0)

    parser.add_argument("--cai_Rx", type=float, default=3.0)
    parser.add_argument("--cai_R", type=float, default=3.0)
    parser.add_argument("--cai_C", type=float, default=1.0)

    parser.add_argument("--fm_Rx", type=float, default=4.0)
    parser.add_argument("--fm_Ry", type=float, default=4.0)
    parser.add_argument("--fm_C", type=float, default=1.0)
    parser.add_argument("--fm_min_eig_floor", type=float, default=1e-6)
    parser.add_argument("--fm_jitter", type=float, default=1e-8)
    parser.add_argument("--fm_lambda_reg", type=float, default=None)
    parser.add_argument("--fm_no_spectral_trimming", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    defaults = ExperimentDefaults(
        p=args.p,
        tau=args.tau,
        sigma_eps=args.sigma_eps,
        n_test=args.n_test,
        reps=args.reps,
        eptr_c0=args.eptr_c0,
        eptr_Rx=args.eptr_Rx,
        eptr_Rtheta=args.eptr_Rtheta,
        sheffet_Rx=args.sheffet_Rx,
        sheffet_Ry=args.sheffet_Ry,
        cai_Rx=args.cai_Rx,
        cai_R=args.cai_R,
        cai_C=args.cai_C,
        fm_Rx=args.fm_Rx,
        fm_Ry=args.fm_Ry,
        fm_C=args.fm_C,
        fm_min_eig_floor=args.fm_min_eig_floor,
        fm_jitter=args.fm_jitter,
        fm_lambda_reg=args.fm_lambda_reg,
        fm_use_spectral_trimming=not args.fm_no_spectral_trimming,
    )

    raw_records = []
    for n in args.n_values:
        for rep in range(args.reps):
            rep_seed = args.seed_base + 1000 * rep + int(n)
            raw_records.extend(
                run_all_methods_one_rep(
                    n=n,
                    epsilon=args.epsilon,
                    defaults=defaults,
                    rep_seed=rep_seed,
                )
            )
            print(f"finished n={n}, rep={rep + 1}/{args.reps}")

    save_csv(outdir / "vary_n_raw.csv", raw_records)

    for metric_key, y_label, stem in METRICS:
        summary = summarize_records(raw_records, x_key="n", metric_key=metric_key)
        save_csv(outdir / f"vary_n_{stem}_summary.csv", summary)
        plot_summary(
            summary,
            x_key="n",
            x_label=r"Sample size $n$",
            y_label=y_label,
            title=f"{y_label} vs. sample size ($\\varepsilon={args.epsilon}$)",
            output_prefix=outdir / f"vary_n_{stem}",
            use_log_x=False,
        )

    print(f"Saved raw results to {outdir / 'vary_n_raw.csv'}")
    for _, _, stem in METRICS:
        print(f"Saved summary results to {outdir / f'vary_n_{stem}_summary.csv'}")
        print(f"Saved plots to {outdir / f'vary_n_{stem}.png'} and {outdir / f'vary_n_{stem}.pdf'}")