"""
Run simulations and compare RBC policy from the trained neural network (learn_rbc)
vs Time Iteration (rbc_TimeIter). Loads a saved NN checkpoint if present; otherwise
trains and saves it so future runs skip training.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from learn_rbc import Params, RBCSolver
from rbc_TimeIter import RBCTISolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Default checkpoint path (same as learn_rbc.save)
CHECKPOINT_PATH = "rbc_nn.pt"
# Simulation length and seed (same for NN and TI)
T_SIM = 200
SIM_SEED = 123


def get_calibration_params():
    """
    Parameter set used for comparison (TI solve + both simulations).
    Edit this to use a different calibration; must lie within the bounds
    used when training the NN (see learn_rbc.Params alpha_bounds, etc.).
    """
    return Params(
        alpha=0.30,
        beta=0.95,
        delta=0.10,
        gamma=2.0,
        rho=0.90,
        sigma_eps=0.02,
    )


def get_nn_solver(train_if_missing: bool = True, device: str = None):
    """
    Return an RBCSolver with trained weights. Load from checkpoint if it exists;
    otherwise train and save, then return.
    """
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(CHECKPOINT_PATH)
    if path.exists():
        return RBCSolver.load(str(path), device=device)
    if not train_if_missing:
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run learn_rbc.py once to train and save, or set train_if_missing=True."
        )
    logger.info("No checkpoint found; training NN...")
    params = Params()
    solver = RBCSolver(params, device=device)
    solver.train(batch_size=2048, epochs=10000)
    solver.save(CHECKPOINT_PATH)
    return solver


def run_comparison(
    params=None,
    train_if_missing: bool = True,
    device: str = None,
    save_plot: str = "rbc_comparison.png",
):
    """
    Run NN and TI at the same calibration, same shock seed, and plot.

    params: Params instance for this calibration. If None, uses get_calibration_params().
    """
    if params is None:
        params = get_calibration_params()
    # ----- NN: load or train -----
    nn_solver = get_nn_solver(train_if_missing=train_if_missing, device=device)
    # ----- TI: solve at same params -----
    ti_solver = RBCTISolver(params)
    policy_ti = ti_solver.solve()
    # ----- Same seed for both simulations; NN simulate at this calibration -----
    np.random.seed(SIM_SEED)
    nn_results = nn_solver.simulate(
        T=T_SIM,
        alpha=params.alpha,
        beta=params.beta,
        delta=params.delta,
        rho=params.rho,
        gamma=params.gamma,
    )
    np.random.seed(SIM_SEED)
    ti_results = ti_solver.simulate(policy_ti, T=T_SIM)
    # ----- Comparison plot -----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    t = np.arange(T_SIM)
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
    plt.suptitle("RBC: Neural network vs Time Iteration (same calibration, same seed)")
    plt.tight_layout()
    plt.savefig(save_plot, dpi=150)
    plt.close()
    logger.info("Saved comparison plot to %s", save_plot)
    return nn_results, ti_results


if __name__ == "__main__":
    # Use calibration from get_calibration_params(); override with e.g.:
    #   run_comparison(params=Params(alpha=0.33, beta=0.97, ...), ...)
    run_comparison(params=get_calibration_params(), train_if_missing=True, save_plot="rbc_comparison.png")
