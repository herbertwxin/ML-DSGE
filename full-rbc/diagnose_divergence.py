"""
Automatic NN-vs-TI divergence diagnostics over random parameter draws.

Outputs (all in full-rbc/):
- divergence_summary.csv: one row per sampled parameter set
- divergence_top_cases.json: top-K most divergent parameter sets + metrics
- divergence_case_XXX_paths.csv: full NN/TI time paths for top-K cases
"""
import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from compare_rbc import FULL_RBC_DIR, CHECKPOINT_PATH, get_nn_solver
from learn_rbc import Params, a_support_from_shock_params, train_rbc_model
from rbc_TimeIter import RBCTISolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

SERIES_KEYS = ("consumption", "capital", "output", "investment", "productivity")
_WORKER_NN_SOLVER = None


def sample_params(rng: np.random.Generator, base: Params) -> Params:
    return Params(
        alpha=float(rng.uniform(*base.alpha_bounds)),
        beta=float(rng.uniform(*base.beta_bounds)),
        delta=float(rng.uniform(*base.delta_bounds)),
        gamma=float(rng.uniform(*base.gamma_bounds)),
        rho=float(rng.uniform(*base.rho_bounds)),
        sigma_eps=float(rng.uniform(*base.sigma_eps_bounds)),
        k_bounds=base.k_bounds,
        A_sigma_mult=base.A_sigma_mult,
        alpha_bounds=base.alpha_bounds,
        beta_bounds=base.beta_bounds,
        delta_bounds=base.delta_bounds,
        rho_bounds=base.rho_bounds,
        gamma_bounds=base.gamma_bounds,
        sigma_eps_bounds=base.sigma_eps_bounds,
    )


def compute_gap_metrics(nn_results: dict, ti_results: dict) -> tuple[dict, float]:
    metrics = {}
    nrmse_values = []
    for key in SERIES_KEYS:
        nn = np.asarray(nn_results[key])
        ti = np.asarray(ti_results[key])
        diff = nn - ti
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        nrmse = rmse / float(np.std(ti) + 1e-10)
        metrics[key] = {"rmse": rmse, "max_abs": max_abs, "nrmse_vs_ti_std": nrmse}
        nrmse_values.append(nrmse)
    score = float(np.max(nrmse_values))
    metrics["aggregate"] = {"mean_nrmse": float(np.mean(nrmse_values)), "max_nrmse": score}
    return metrics, score


def save_path_table(nn_results: dict, ti_results: dict, output_csv: Path) -> None:
    t = np.arange(len(nn_results["consumption"]))
    cols = [t]
    names = ["t"]
    for key in SERIES_KEYS:
        nn = np.asarray(nn_results[key])
        ti = np.asarray(ti_results[key])
        cols.extend([nn, ti, nn - ti])
        names.extend([f"{key}_nn", f"{key}_ti", f"{key}_diff"])
    table = np.column_stack(cols)
    np.savetxt(output_csv, table, delimiter=",", header=",".join(names), comments="")


def save_comparison_plot(nn_results: dict, ti_results: dict, output_png: Path, params: Params) -> None:
    t = np.arange(len(nn_results["consumption"]))
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes[0, 0].plot(t, nn_results["consumption"], label="NN", linewidth=1.5)
    axes[0, 0].plot(t, ti_results["consumption"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[0, 0].set_title("Consumption")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("t")

    axes[0, 1].plot(t, nn_results["capital"], label="NN", linewidth=1.5)
    axes[0, 1].plot(t, ti_results["capital"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[0, 1].set_title("Capital")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("t")

    axes[1, 0].plot(t, nn_results["output"], label="NN", linewidth=1.5)
    axes[1, 0].plot(t, ti_results["output"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[1, 0].set_title("Output")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("t")

    axes[1, 1].plot(t, nn_results["investment"], label="NN", linewidth=1.5)
    axes[1, 1].plot(t, ti_results["investment"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[1, 1].set_title("Investment")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("t")

    axes[2, 0].plot(t, nn_results["productivity"], label="NN", linewidth=1.5)
    axes[2, 0].plot(t, ti_results["productivity"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[2, 0].set_title("TFP (productivity)")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel("t")
    axes[2, 1].axis("off")
    param_text = (
        "Parameter set\n"
        f"alpha      = {params.alpha:.4f}\n"
        f"beta       = {params.beta:.4f}\n"
        f"delta      = {params.delta:.4f}\n"
        f"rho        = {params.rho:.4f}\n"
        f"gamma      = {params.gamma:.4f}\n"
        f"sigma_eps  = {params.sigma_eps:.4f}\n"
        f"A_sigma_mult = {params.A_sigma_mult:.2f}"
    )
    axes[2, 1].text(
        0.02,
        0.98,
        param_text,
        transform=axes[2, 1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    plt.suptitle("RBC: Neural network vs Time Iteration (same calibration, same seed)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def run_case(nn_solver, params: Params, T: int, seed: int) -> tuple[dict, dict, dict]:
    ti_solver = RBCTISolver(params)
    policy_ti = ti_solver.solve()

    np.random.seed(seed)
    nn_results = nn_solver.simulate(
        T=T,
        alpha=params.alpha,
        beta=params.beta,
        delta=params.delta,
        rho=params.rho,
        gamma=params.gamma,
        sigma_eps=params.sigma_eps,
    )
    np.random.seed(seed)
    ti_results = ti_solver.simulate(policy_ti, T=T)

    metrics, score = compute_gap_metrics(nn_results, ti_results)

    # Diagnostics to localize likely failure modes
    k_ss_sim = nn_solver._steady_state_batch(
        torch.tensor(params.alpha), torch.tensor(params.beta), torch.tensor(params.delta)
    )[0].item()
    nn_k_low = params.k_bounds[0] * k_ss_sim
    nn_k_high = params.k_bounds[1] * k_ss_sim
    nn_k = np.asarray(nn_results["capital"])
    nn_k_oob_frac = float(np.mean((nn_k < nn_k_low) | (nn_k > nn_k_high)))

    a_low, a_high = a_support_from_shock_params(params.rho, params.sigma_eps, params.A_sigma_mult, 1.0)
    nn_A = np.asarray(nn_results["productivity"])
    nn_a_oob_frac = float(np.mean((nn_A < a_low) | (nn_A > a_high)))

    ti_k = np.asarray(ti_results["capital"])
    ti_A = np.asarray(ti_results["productivity"])
    ti_k_oob_frac = float(np.mean((ti_k < ti_solver.k_min) | (ti_k > ti_solver.k_max)))
    ti_a_oob_frac = float(np.mean((ti_A < ti_solver.A_min) | (ti_A > ti_solver.A_max)))

    summary_row = {
        "score_max_nrmse": score,
        "alpha": params.alpha,
        "beta": params.beta,
        "delta": params.delta,
        "rho": params.rho,
        "gamma": params.gamma,
        "sigma_eps": params.sigma_eps,
        "ti_k_oob_frac": ti_k_oob_frac,
        "ti_a_oob_frac": ti_a_oob_frac,
        "nn_k_oob_frac": nn_k_oob_frac,
        "nn_a_oob_frac": nn_a_oob_frac,
        "mean_nrmse": metrics["aggregate"]["mean_nrmse"],
        "nrmse_c": metrics["consumption"]["nrmse_vs_ti_std"],
        "nrmse_k": metrics["capital"]["nrmse_vs_ti_std"],
        "nrmse_y": metrics["output"]["nrmse_vs_ti_std"],
        "nrmse_i": metrics["investment"]["nrmse_vs_ti_std"],
        "nrmse_A": metrics["productivity"]["nrmse_vs_ti_std"],
    }
    detail = {
        "params": asdict(params),
        "metrics": metrics,
        "diagnostics": {
            "ti_k_oob_frac": ti_k_oob_frac,
            "ti_a_oob_frac": ti_a_oob_frac,
            "nn_k_oob_frac": nn_k_oob_frac,
            "nn_a_oob_frac": nn_a_oob_frac,
        },
    }
    return summary_row, detail, {"nn": nn_results, "ti": ti_results}


def run_case_payload(payload: dict) -> tuple[int, dict, dict, dict]:
    """
    Worker-safe wrapper: receives plain objects, recreates Params, loads NN lazily once per process.
    """
    global _WORKER_NN_SOLVER
    if _WORKER_NN_SOLVER is None:
        _WORKER_NN_SOLVER = get_nn_solver(train_if_missing=False)
    case_idx = payload["case_idx"]
    params = Params(**payload["params"])
    summary, detail, paths = run_case(
        _WORKER_NN_SOLVER,
        params=params,
        T=payload["T"],
        seed=payload["seed"],
    )
    return case_idx, summary, detail, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=40, help="Number of random parameter draws.")
    parser.add_argument("--top-k", type=int, default=5, help="Save full paths for top-K divergent cases.")
    parser.add_argument("--T", type=int, default=200, help="Simulation length per case.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for draw reproducibility.")
    parser.add_argument("--train-if-missing", action="store_true", help="Train NN if no checkpoint exists.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of parallel worker processes for case evaluations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation",
        help="Folder under full-rbc to save all outputs (plots/csv/json).",
    )
    args = parser.parse_args()

    output_dir = (FULL_RBC_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving simulation outputs to %s", output_dir)

    base_params = Params()
    rng = np.random.default_rng(args.seed)
    # Ensure checkpoint exists before spawning workers (avoids parallel retraining race).
    if (not CHECKPOINT_PATH.exists()) and args.train_if_missing:
        logger.info("No checkpoint found. Running dedicated training script logic first...")
        train_rbc_model()
    nn_solver = get_nn_solver(train_if_missing=False)

    payloads = []
    sampled_params = []
    for i in range(args.n_cases):
        params_i = sample_params(rng, base_params)
        sampled_params.append(params_i)
        payloads.append(
            {
                "case_idx": i,
                "params": asdict(params_i),
                "T": args.T,
                "seed": args.seed + i,
            }
        )

    summaries = []
    details = []
    paths_cache = []
    if args.n_workers <= 1:
        for i, params_i in enumerate(sampled_params):
            summary_i, detail_i, paths_i = run_case(nn_solver, params_i, args.T, args.seed + i)
            summaries.append(summary_i)
            details.append(detail_i)
            paths_cache.append(paths_i)
            logger.info(
                "Case %d/%d score %.4f | alpha=%.3f beta=%.3f delta=%.3f rho=%.3f gamma=%.3f sigma=%.3f",
                i + 1,
                args.n_cases,
                summary_i["score_max_nrmse"],
                params_i.alpha,
                params_i.beta,
                params_i.delta,
                params_i.rho,
                params_i.gamma,
                params_i.sigma_eps,
            )
    else:
        logger.info("Running %d cases with %d worker processes...", args.n_cases, args.n_workers)
        summaries = [None] * args.n_cases
        details = [None] * args.n_cases
        paths_cache = [None] * args.n_cases
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            for case_idx, summary_i, detail_i, paths_i in executor.map(run_case_payload, payloads):
                params_i = sampled_params[case_idx]
                summaries[case_idx] = summary_i
                details[case_idx] = detail_i
                paths_cache[case_idx] = paths_i
                logger.info(
                    "Case %d/%d score %.4f | alpha=%.3f beta=%.3f delta=%.3f rho=%.3f gamma=%.3f sigma=%.3f",
                    case_idx + 1,
                    args.n_cases,
                    summary_i["score_max_nrmse"],
                    params_i.alpha,
                    params_i.beta,
                    params_i.delta,
                    params_i.rho,
                    params_i.gamma,
                    params_i.sigma_eps,
                )

    order = np.argsort([row["score_max_nrmse"] for row in summaries])[::-1]
    top_k = min(args.top_k, len(order))

    # Save full paths and plots for all evaluated cases (ordered by divergence rank).
    for rank, idx in enumerate(order, start=1):
        case_paths_csv = output_dir / f"divergence_case_{rank:03d}_paths.csv"
        case_plot_png = output_dir / f"divergence_case_{rank:03d}_plot.png"
        save_path_table(paths_cache[idx]["nn"], paths_cache[idx]["ti"], case_paths_csv)
        case_params = Params(**details[idx]["params"])
        save_comparison_plot(paths_cache[idx]["nn"], paths_cache[idx]["ti"], case_plot_png, case_params)

    summary_path = output_dir / "divergence_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for idx in order:
            writer.writerow(summaries[idx])
    logger.info("Saved case summary to %s", summary_path)

    top_cases = []
    for rank, idx in enumerate(order[:top_k], start=1):
        path_file = output_dir / f"divergence_case_{rank:03d}_paths.csv"
        plot_file = output_dir / f"divergence_case_{rank:03d}_plot.png"
        top_cases.append(
            {
                "rank": rank,
                "summary": summaries[idx],
                "detail": details[idx],
                "paths_file": str(path_file),
                "plot_file": str(plot_file),
            }
        )

    top_path = output_dir / "divergence_top_cases.json"
    top_path.write_text(json.dumps(top_cases, indent=2))
    logger.info("Saved top-case diagnostics to %s", top_path)


if __name__ == "__main__":
    main()
