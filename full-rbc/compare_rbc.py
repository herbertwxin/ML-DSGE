"""
Run simulations and compare RBC policy from the trained neural network (learn_rbc)
vs Time Iteration (rbc_TimeIter). Loads a saved NN checkpoint if present; otherwise
trains and saves it so future runs skip training.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from learn_rbc import Params, RBCSolver, get_device
from rbc_TimeIter import RBCTISolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# All paths under full-rbc (script directory) so inputs/outputs stay in this folder
FULL_RBC_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = FULL_RBC_DIR / "rbc_nn.pt"
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
        delta=0.20,
        gamma=2.0,
        rho=0.70,
        sigma_eps=0.02,
    )


def get_nn_solver(train_if_missing: bool = True, device: str = None):
    """
    Return an RBCSolver with trained weights. Load from checkpoint if it exists;
    otherwise train and save, then return. Checkpoint is read from full-rbc folder.
    """
    if device is None:
        device = get_device()
    path = CHECKPOINT_PATH
    if path.exists():
        return RBCSolver.load(str(path), device=device)
    if not train_if_missing:
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run learn_rbc.py once to train and save, or set train_if_missing=True."
        )
    logger.info("No checkpoint found; training NN...")
    params = Params()
    solver = RBCSolver(params, device=device)
    solver.train(batch_size=2048, epochs=50000)
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
    # ----- NN: load or train (weights only; calibration is applied at simulate time) -----
    nn_solver = get_nn_solver(train_if_missing=train_if_missing, device=device)
    # ----- TI: solve at same params -----
    ti_solver = RBCTISolver(params)
    policy_ti = ti_solver.solve()
    # ----- Same seed for both simulations; NN evaluated at this calibration (all params fed in) -----
    np.random.seed(SIM_SEED)
    nn_results = nn_solver.simulate(
        T=T_SIM,
        alpha=params.alpha,
        beta=params.beta,
        delta=params.delta,
        rho=params.rho,
        gamma=params.gamma,
        sigma_eps=params.sigma_eps,
    )
    np.random.seed(SIM_SEED)
    ti_results = ti_solver.simulate(policy_ti, T=T_SIM)
    # ----- Comparison plot -----
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
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
    # TFP (productivity): same shock seed => same path for NN and TI
    axes[2, 0].plot(t, nn_results["productivity"], label="NN", linewidth=1.5)
    axes[2, 0].plot(t, ti_results["productivity"], label="TI", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[2, 0].set_title("TFP (productivity)")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel("t")
    axes[2, 1].set_visible(False)
    plt.suptitle("RBC: Neural network vs Time Iteration (same calibration, same seed)")
    plt.tight_layout()
    plot_path = Path(save_plot)
    if not plot_path.is_absolute():
        plot_path = FULL_RBC_DIR / plot_path
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Saved comparison plot to %s", plot_path)
    return nn_results, ti_results


if __name__ == "__main__":
    # Use calibration from get_calibration_params(); override with e.g.:
    #   run_comparison(params=Params(alpha=0.33, beta=0.97, ...), ...)
    run_comparison(
        params=get_calibration_params(),
        train_if_missing=True,
        save_plot=str(FULL_RBC_DIR / "rbc_comparison.png"),
    )
