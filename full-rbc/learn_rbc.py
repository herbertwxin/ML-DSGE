"""
Learn RBC (Real Business Cycle) policy with a neural network over a WIDE RANGE
of structural parameters. The NN is trained to satisfy the Euler equation
across many (alpha, beta, delta, gamma, rho) so it approximates the whole
RBC model, not just one calibration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_device() -> str:
    """
    Return the best available device: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Use this so training runs on GPU when available, including on Apple devices.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Params:
    """Model parameters for the RBC model. Defaults are reference values; we train over ranges."""
    alpha: float = 0.30      # capital share
    beta: float = 0.95       # discount factor
    delta: float = 0.1       # depreciation rate
    gamma: float = 2.0       # risk aversion
    rho: float = 0.90        # persistence of productivity shock
    sigma_eps: float = 0.02  # std dev of shock innovation (fixed in training)

    # Bounds for state space: k and A as fraction of steady state or level
    k_bounds: tuple = (0.5, 1.5)   # k as fraction of steady-state capital
    A_bounds: tuple = (0.5, 1.5)   # productivity level

    # Bounds for STRUCTURAL PARAMETERS (NN learns over this whole space)
    alpha_bounds: tuple = (0.20, 0.45)   # capital share
    beta_bounds: tuple = (0.90, 0.99)    # discount factor
    delta_bounds: tuple = (0.02, 0.15)   # depreciation rate
    rho_bounds: tuple = (0.85, 0.99)      # persistence of productivity
    gamma_bounds: tuple = (0.5, 4.0)     # risk aversion

class RBCNet(nn.Module):
    """Neural network approximating the policy (consumption fraction) for the RBC model.
    Inputs: (k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm).
    Output: fraction of current resources consumed (sigmoid → [0,1]).
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, output_bias: float = 0.0):
        super(RBCNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        output_layer = nn.Linear(prev_dim, output_dim)
        with torch.no_grad():
            output_layer.bias.fill_(output_bias)
        layers.append(output_layer)
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RBCSolver:
    """
    Trains a single NN to approximate the RBC policy over a wide range of
    (alpha, beta, delta, rho, gamma). Steady state is computed per-sample
    when parameters vary.
    """

    def __init__(self, params: Params, device: str = "cpu"):
        self.p = params
        self.device = torch.device(device)

        # Reference steady state (at default params) for output-bias initialization only
        self.k_ss, self.c_ss, self.y_ss, self.A_ss = self._steady_state(
            self.p.alpha, self.p.beta, self.p.delta
        )
        logger.info(f"Reference steady-state capital (A=1): {self.k_ss:.3f}")

        # Initial bias so policy starts near steady-state consumption share
        res_ss_init = self.y_ss + (1.0 - self.p.delta) * self.k_ss
        frac_ss_init = self.c_ss / res_ss_init
        init_bias = np.log(frac_ss_init / (1.0 - frac_ss_init))
        logger.info(f"Output bias init: {init_bias:.3f} (SS frac: {frac_ss_init:.3f})")

        # 7 inputs: k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm
        self.model = RBCNet(7, [256, 128, 64, 32], 1, output_bias=init_bias).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

        # Hermite-Gauss quadrature for E[·] in Euler equation
        self.n_quad = 7
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        self.z_nodes = torch.tensor(nodes * np.sqrt(2), dtype=torch.float32, device=self.device)
        self.z_weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32, device=self.device)

    def _steady_state(self, alpha: float, beta: float, delta: float, A_ss: float = 1.0):
        """Steady state for given (alpha, beta, delta), A=1."""
        term = (1.0 / beta - (1.0 - delta)) / (alpha * A_ss)
        k_ss = term ** (1.0 / (alpha - 1.0))
        y_ss = A_ss * k_ss ** alpha
        c_ss = y_ss - delta * k_ss
        return k_ss, c_ss, y_ss, A_ss

    def _steady_state_batch(self, alpha: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor):
        """Vectorized steady-state capital for batches (for per-sample k bounds)."""
        A_ss = 1.0
        term = (1.0 / beta - (1.0 - delta)) / (alpha * A_ss)
        k_ss = term ** (1.0 / (alpha - 1.0))
        return k_ss

    def normalize(self, x, x_low, x_high):
        return (x - x_low) / (x_high - x_low)

    def denormalize(self, x, x_low, x_high):
        return x * (x_high - x_low) + x_low

    def sample_batch(self, batch_size: int):
        """
        Sample (k, A, alpha, beta, delta, rho, gamma) over the full parameter range.
        k is normalized in [0,1] and will be interpreted as fraction of steady-state
        capital in compute_residuals (per-sample k_ss). A and structural params
        are normalized to [0,1] within their bounds.
        """
        p = self.p

        # States: k and A (k in [0,1] → fraction of SS; A will be scaled in residuals)
        k_batch = torch.rand(batch_size, device=self.device)
        alpha_batch = torch.rand(batch_size, device=self.device)
        beta_batch = torch.rand(batch_size, device=self.device)
        delta_batch = torch.rand(batch_size, device=self.device)
        rho_batch = torch.rand(batch_size, device=self.device)
        gamma_batch = torch.rand(batch_size, device=self.device)

        # A: lognormal with persistence-dependent variance
        rho_val = self.denormalize(rho_batch, p.rho_bounds[0], p.rho_bounds[1])
        sigma_stat = p.sigma_eps / torch.sqrt(1.0 - rho_val**2)
        A = torch.exp(sigma_stat * torch.randn(batch_size, device=self.device))
        A_batch = self.normalize(A, p.A_bounds[0], p.A_bounds[1])

        # Order: k, A, alpha, beta, delta, rho, gamma
        inputs = torch.stack(
            [k_batch, A_batch, alpha_batch, beta_batch, delta_batch, rho_batch, gamma_batch],
            dim=1,
        )
        return inputs

    def compute_residuals(self, inputs: torch.Tensor):
        """
        Euler residuals over a batch. Parameters (alpha, beta, delta, rho, gamma)
        vary per sample; k is scaled by per-sample steady-state capital so the
        NN sees a consistent state space across the parameter range.
        """
        p = self.p

        # Unpack normalized inputs (7 dims)
        k_norm = inputs[:, 0]
        A_norm = inputs[:, 1]
        alpha_norm = inputs[:, 2]
        beta_norm = inputs[:, 3]
        delta_norm = inputs[:, 4]
        rho_norm = inputs[:, 5]
        gamma_norm = inputs[:, 6]

        # Denormalize structural parameters (vary per sample)
        alpha = self.denormalize(alpha_norm, p.alpha_bounds[0], p.alpha_bounds[1])
        beta = self.denormalize(beta_norm, p.beta_bounds[0], p.beta_bounds[1])
        delta = self.denormalize(delta_norm, p.delta_bounds[0], p.delta_bounds[1])
        rho = self.denormalize(rho_norm, p.rho_bounds[0], p.rho_bounds[1])
        gamma = self.denormalize(gamma_norm, p.gamma_bounds[0], p.gamma_bounds[1])

        # Per-sample steady-state capital so k has consistent meaning across params
        k_ss_batch = self._steady_state_batch(alpha, beta, delta)
        k_low = p.k_bounds[0] * k_ss_batch
        k_high = p.k_bounds[1] * k_ss_batch
        k = self.denormalize(k_norm, k_low, k_high)
        A = self.denormalize(A_norm, p.A_bounds[0], p.A_bounds[1])

        # Policy: fraction of resources consumed
        frac = self.model(inputs).squeeze()

        # Current resources (use per-sample alpha, delta)
        resources = A * (k ** alpha) + (1.0 - delta) * k
        c = frac * resources
        k_next = resources - c
        k_next = torch.clamp(k_next, k_low, k_high)
        c = resources - k_next
        c = torch.clamp(c, min=1e-6)

        mu = c ** (-gamma)
        expected_rhs = torch.zeros_like(mu)
        k_next_norm = self.normalize(k_next, k_low, k_high)

        for i in range(self.n_quad):
            eps = self.z_nodes[i]
            weight = self.z_weights[i]
            log_A_next = rho * torch.log(A.clamp(min=1e-8)) + p.sigma_eps * eps
            A_next = torch.exp(log_A_next).clamp(p.A_bounds[0], p.A_bounds[1])
            A_next_norm = self.normalize(A_next, p.A_bounds[0], p.A_bounds[1])

            inputs_next = torch.stack([
                k_next_norm,
                A_next_norm,
                alpha_norm,
                beta_norm,
                delta_norm,
                rho_norm,
                gamma_norm,
            ], dim=1)

            frac_next = self.model(inputs_next).squeeze()
            resources_next = A_next * (k_next ** alpha) + (1.0 - delta) * k_next
            k_next_next = (1.0 - frac_next) * resources_next
            k_next_next = torch.clamp(k_next_next, k_low, k_high)
            c_next = (resources_next - k_next_next).clamp(min=1e-6)
            mu_next = c_next ** (-gamma)
            R_next = alpha * A_next * (k_next ** (alpha - 1.0)) + (1.0 - delta)
            expected_rhs += weight * (beta * mu_next * R_next)

        return expected_rhs - mu

    def train(self, batch_size: int = 2048, epochs: int = 10000):
        """Train on random (k, A, alpha, beta, delta, rho, gamma) so the NN learns the RBC model over the full parameter space."""
        self.model.train()
        losses = []
        logger.info(f"Training over wide parameter range on {self.device}...")
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            
            inputs = self.sample_batch(batch_size)
            residuals = self.compute_residuals(inputs)
            
            loss = torch.mean(residuals ** 2)
            loss.backward()
            
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, MSE: {loss.item():.3e}")
                
        return losses

    def simulate(
        self,
        T: int = 200,
        k0: float = None,
        A0: float = None,
        alpha: float = None,
        beta: float = None,
        delta: float = None,
        rho: float = None,
        gamma: float = None,
        sigma_eps: float = None,
    ) -> dict:
        """
        Simulate the economy at a given parameter point. Uses solver default
        params for any argument left as None. After training over a wide range,
        you can simulate at any (alpha, beta, delta, rho, gamma, sigma_eps) within bounds.
        """
        self.model.eval()
        p = self.p
        alpha = alpha if alpha is not None else p.alpha
        beta = beta if beta is not None else p.beta
        delta = delta if delta is not None else p.delta
        rho = rho if rho is not None else p.rho
        gamma = gamma if gamma is not None else p.gamma
        sigma_eps = sigma_eps if sigma_eps is not None else p.sigma_eps

        k_ss_sim, c_ss_sim, y_ss_sim, _ = self._steady_state(alpha, beta, delta)
        if k0 is None:
            k0 = k_ss_sim
        if A0 is None:
            A0 = self.A_ss

        k_series = np.zeros(T + 1)
        A_series = np.zeros(T + 1)
        c_series = np.zeros(T)
        y_series = np.zeros(T)
        i_series = np.zeros(T)
        k_series[0] = k0
        A_series[0] = A0
        eps_series = np.random.randn(T)

        k_low = p.k_bounds[0] * k_ss_sim
        k_high = p.k_bounds[1] * k_ss_sim
        k_norm = lambda k: self.normalize(torch.tensor(k, dtype=torch.float32), k_low, k_high)
        A_norm = lambda A: self.normalize(torch.tensor(A, dtype=torch.float32), p.A_bounds[0], p.A_bounds[1])
        alpha_n = self.normalize(torch.tensor(alpha), p.alpha_bounds[0], p.alpha_bounds[1])
        beta_n = self.normalize(torch.tensor(beta), p.beta_bounds[0], p.beta_bounds[1])
        delta_n = self.normalize(torch.tensor(delta), p.delta_bounds[0], p.delta_bounds[1])
        rho_n = self.normalize(torch.tensor(rho), p.rho_bounds[0], p.rho_bounds[1])
        gamma_n = self.normalize(torch.tensor(gamma), p.gamma_bounds[0], p.gamma_bounds[1])

        with torch.no_grad():
            for t in range(T):
                k, A = k_series[t], A_series[t]
                state = torch.stack([
                    k_norm(k), A_norm(A), alpha_n, beta_n, delta_n, rho_n, gamma_n,
                ]).unsqueeze(0).to(self.device)
                frac = self.model(state).item()
                y = A * k ** alpha
                y_series[t] = y
                resources = y + (1.0 - delta) * k
                c = frac * resources
                c_series[t] = c
                i_series[t] = y - c
                k_series[t + 1] = max((1.0 - delta) * k + y - c, 1e-6)
                log_A_next = rho * np.log(max(A, 1e-8)) + sigma_eps * eps_series[t]
                A_series[t + 1] = np.exp(log_A_next)

        return {
            "capital": k_series[:T],
            "productivity": A_series[:T],
            "consumption": c_series,
            "output": y_series,
            "investment": i_series,
        }

    def save(self, path: str = "rbc_nn.pt") -> None:
        """Save trained model and Params so we can load without retraining."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "params": asdict(self.p)},
            path,
        )
        logger.info(f"Saved model and params to {path}")

    @staticmethod
    def load(path: str, device: str = None) -> "RBCSolver":
        """Load solver from checkpoint (no training)."""
        if device is None:
            device = get_device()
        d = torch.load(path, map_location=device)
        params = Params(**d["params"])
        solver = RBCSolver(params, device=device)
        solver.model.load_state_dict(d["state_dict"])
        logger.info(f"Loaded model from {path}")
        return solver


if __name__ == "__main__":
    # Train one NN to approximate the RBC policy over the full parameter space
    device = get_device()
    logger.info("Using device: %s", device)
    params = Params()
    solver = RBCSolver(params, device=device)
    losses = solver.train(batch_size=2048, epochs=50000)

    # Simulate at default calibration
    sim_default = solver.simulate(T=200)
    logger.info("Simulation at default params done.")

    # Simulate at a different calibration (to check generalization)
    sim_alt = solver.simulate(T=200, alpha=0.33, beta=0.97, delta=0.08, rho=0.95, gamma=1.5)
    logger.info("Simulation at alternate params done.")

    # Optional: plot training loss (output in same folder as script)
    out_dir = Path(__file__).resolve().parent
    loss_plot = out_dir / "learn_rbc_loss.png"
    plt.figure(figsize=(6, 4))
    plt.semilogy(losses, alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (Euler residual)")
    plt.title("RBC NN training over wide parameter range")
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=150)
    plt.close()
    logger.info("Saved %s", loss_plot)
    # Save checkpoint in full-rbc so compare_rbc.py (and future runs) can skip training
    solver.save(str(out_dir / "rbc_nn.pt"))