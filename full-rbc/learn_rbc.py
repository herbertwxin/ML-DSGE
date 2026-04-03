"""
Learn RBC (Real Business Cycle) policy with a neural network over a WIDE RANGE
of structural parameters. Productivity A is normalized using bounds that depend
on (rho, sigma_eps) via the unconditional scale of log TFP: sigma_stat = sigma_eps / sqrt(1-rho^2).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ThreadPoolExecutor
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

    # Bounds for state space: k as fraction of steady-state capital
    k_bounds: tuple = (0.5, 1.5)   # k as fraction of steady-state capital
    # A_bounds: tuple = (0.5, 1.5)

    # How wide the NN's A normalization box is: log A in [-n, +n] * sigma_stat, sigma_stat = sigma_eps/sqrt(1-rho^2)
    A_sigma_mult: float = 3.0

    # Bounds for STRUCTURAL PARAMETERS (NN learns over this whole space)
    alpha_bounds: tuple = (0.20, 0.45)   # capital share
    beta_bounds: tuple = (0.90, 0.99)    # discount factor
    delta_bounds: tuple = (0.02, 0.15)   # depreciation rate
    rho_bounds: tuple = (0.85, 0.99)      # persistence of productivity
    gamma_bounds: tuple = (0.5, 4.0)     # risk aversion
    sigma_eps_bounds: tuple = (0.005, 0.05)  # shock innovation std (NN input + A scaling)

    # Oversampling settings for hard region (high beta, low delta) in training batches.
    hard_region_prob: float = 0.0
    hard_beta_low_norm: float = 0.85   # beta_norm sampled from [hard_beta_low_norm, 1]
    hard_delta_high_norm: float = 0.20 # delta_norm sampled from [0, hard_delta_high_norm]


def a_support_from_shock_params(
    rho: float,
    sigma_eps: float,
    a_sigma_mult: float,
    a_ss: float = 1.0,
) -> tuple[float, float]:
    """
    Physical productivity support [A_low, A_high] = exp(± n σ_stat) × A_ss,
    σ_stat = σ_ε / sqrt(1-ρ²). Same rule for NN state normalization and TI spline grid.
    """
    one_m = max(1e-4, 1.0 - rho * rho)
    sigma_stat = sigma_eps / np.sqrt(one_m)
    w = a_sigma_mult * sigma_stat
    a_low = float(np.exp(-w) * a_ss)
    a_high = float(np.exp(w) * a_ss)
    return a_low, max(a_high, a_low + 1e-6)


class RBCNet(nn.Module):
    """Neural network approximating the policy (consumption fraction) for the RBC model.
    Inputs: (k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm).
    A_norm is vs A_low, A_high from stationary log-TFP scale given (rho, sigma_eps).
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
    Trains a single NN over (alpha, beta, delta, rho, gamma, sigma_eps).
    Productivity is scaled with bounds A_low, A_high = exp(± n * sigma_stat),
    sigma_stat = sigma_eps / sqrt(1 - rho^2), so A_norm is comparable across shock processes.
    """

    def __init__(self, params: Params, device: str = "cpu"):
        self.p = params
        self.device = torch.device(device)

        # Reference steady state (at default params) for output-bias initialization only
        _a = torch.tensor(self.p.alpha); _b = torch.tensor(self.p.beta); _d = torch.tensor(self.p.delta)
        self.k_ss, self.c_ss, self.y_ss = (x.item() for x in self._steady_state_batch(_a, _b, _d))
        self.A_ss = 1.0
        logger.info(f"Reference steady-state capital (A=1): {self.k_ss:.3f}")

        # Initial bias so policy starts near steady-state consumption share
        res_ss_init = self.y_ss + (1.0 - self.p.delta) * self.k_ss
        frac_ss_init = self.c_ss / res_ss_init
        init_bias = np.log(frac_ss_init / (1.0 - frac_ss_init))
        logger.info(f"Output bias init: {init_bias:.3f} (SS frac: {frac_ss_init:.3f})")

        # 8 inputs: k_norm, A_norm, ..., sigma_eps_norm (A_norm uses (rho, sigma_eps)-dependent bounds)
        self.model = RBCNet(8, [512, 256, 128, 64, 32, 16], 1, output_bias=init_bias).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

        # Hermite-Gauss quadrature for E[·] in Euler equation
        self.n_quad = 7
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        self.z_nodes = torch.tensor(nodes * np.sqrt(2), dtype=torch.float32, device=self.device)
        self.z_weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32, device=self.device)

    def _steady_state_batch(self, alpha: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor):
        """Vectorized steady state (A=1) for batches. Returns (k_ss, c_ss, y_ss)."""
        term = (1.0 / beta - (1.0 - delta)) / alpha
        k_ss = term ** (1.0 / (alpha - 1.0))
        y_ss = k_ss ** alpha
        c_ss = y_ss - delta * k_ss
        return k_ss, c_ss, y_ss

    def normalize(self, x, x_low, x_high):
        return (x - x_low) / (x_high - x_low)

    def denormalize(self, x, x_low, x_high):
        return x * (x_high - x_low) + x_low

    def _A_bounds_tensors(self, rho: torch.Tensor, sigma_eps: torch.Tensor):
        """Symmetric A bounds from unconditional log-TFP scale: Var(log A) = sigma_eps^2/(1-rho^2)."""
        n = self.p.A_sigma_mult
        one_m = torch.clamp(1.0 - rho * rho, min=1e-4)
        sigma_stat = sigma_eps / torch.sqrt(one_m)
        w = n * sigma_stat
        A_low = torch.exp(-w)
        A_high = torch.exp(w)
        A_high = torch.maximum(A_high, A_low + 1e-6)
        return A_low, A_high

    def _A_bounds_numpy(self, rho: float, sigma_eps: float) -> tuple[float, float]:
        return a_support_from_shock_params(rho, sigma_eps, self.p.A_sigma_mult, self.A_ss)

    def _sample_params_numpy(self, rng: np.random.Generator) -> Params:
        """Draw one parameter set uniformly from training bounds."""
        p = self.p
        return Params(
            alpha=float(rng.uniform(*p.alpha_bounds)),
            beta=float(rng.uniform(*p.beta_bounds)),
            delta=float(rng.uniform(*p.delta_bounds)),
            gamma=float(rng.uniform(*p.gamma_bounds)),
            rho=float(rng.uniform(*p.rho_bounds)),
            sigma_eps=float(rng.uniform(*p.sigma_eps_bounds)),
            k_bounds=p.k_bounds,
            A_sigma_mult=p.A_sigma_mult,
            alpha_bounds=p.alpha_bounds,
            beta_bounds=p.beta_bounds,
            delta_bounds=p.delta_bounds,
            rho_bounds=p.rho_bounds,
            gamma_bounds=p.gamma_bounds,
            sigma_eps_bounds=p.sigma_eps_bounds,
        )

    @staticmethod
    def _gap_metrics(nn_results: dict, ti_results: dict) -> dict:
        keys = ("consumption", "capital", "output", "investment")
        by_var = {}
        nrmse_values = []
        level_ratio_values = {}
        for k in keys:
            nn = np.asarray(nn_results[k])
            ti = np.asarray(ti_results[k])
            rmse = float(np.sqrt(np.mean((nn - ti) ** 2)))
            nrmse = rmse / float(np.std(ti) + 1e-10)
            lvl_ratio = float((np.mean(nn) + 1e-10) / (np.mean(ti) + 1e-10))
            by_var[k] = {"rmse": rmse, "nrmse": nrmse, "level_ratio": lvl_ratio}
            nrmse_values.append(nrmse)
            level_ratio_values[k] = lvl_ratio
        by_var["aggregate"] = {
            "mean_nrmse": float(np.mean(nrmse_values)),
            "max_nrmse": float(np.max(nrmse_values)),
        }
        by_var["level_ratio"] = level_ratio_values
        return by_var

    def _build_validation_panel(self, n_cases: int, seed: int):
        """
        Build a fixed validation panel. TI policies are solved once and reused,
        so diagnostics remain deterministic and cheap at each evaluation.
        """
        if n_cases <= 0:
            return []
        from rbc_TimeIter import RBCTISolver

        rng = np.random.default_rng(seed)
        params_list = [self._sample_params_numpy(rng) for _ in range(n_cases)]

        def _solve_ti_item(item):
            idx, p_i = item
            ti_solver = RBCTISolver(p_i)
            ti_policy = ti_solver.solve()
            return idx, p_i, ti_solver, ti_policy

        panel = [None] * n_cases
        workers = min(8, n_cases)
        if workers > 1:
            logger.info("Building validation TI panel in parallel (%d workers)...", workers)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for idx, p_i, ti_solver, ti_policy in ex.map(_solve_ti_item, list(enumerate(params_list))):
                    panel[idx] = {
                        "params": p_i,
                        "ti_solver": ti_solver,
                        "ti_policy": ti_policy,
                        "seed": seed + idx,
                    }
        else:
            for idx, p_i in enumerate(params_list):
                ti_solver = RBCTISolver(p_i)
                ti_policy = ti_solver.solve()
                panel[idx] = {"params": p_i, "ti_solver": ti_solver, "ti_policy": ti_policy, "seed": seed + idx}
        return panel

    def _evaluate_validation_panel(self, panel, T: int) -> dict:
        if not panel:
            return {}
        metrics_list = []
        prev_mode = self.model.training
        self.model.eval()
        for item in panel:
            p_i = item["params"]
            ti_solver = item["ti_solver"]
            ti_policy = item["ti_policy"]
            seed = item["seed"]
            np.random.seed(seed)
            nn_res = self.simulate(
                T=T,
                alpha=p_i.alpha,
                beta=p_i.beta,
                delta=p_i.delta,
                rho=p_i.rho,
                gamma=p_i.gamma,
                sigma_eps=p_i.sigma_eps,
            )
            np.random.seed(seed)
            ti_res = ti_solver.simulate(ti_policy, T=T)
            metrics_list.append(self._gap_metrics(nn_res, ti_res))
        if prev_mode:
            self.model.train()

        keys = ("consumption", "capital", "output", "investment")
        avg = {"aggregate": {}, "level_ratio": {}}
        for k in keys:
            avg[k] = {
                "rmse": float(np.mean([m[k]["rmse"] for m in metrics_list])),
                "nrmse": float(np.mean([m[k]["nrmse"] for m in metrics_list])),
                "level_ratio": float(np.mean([m[k]["level_ratio"] for m in metrics_list])),
            }
            avg["level_ratio"][k] = avg[k]["level_ratio"]
        avg["aggregate"] = {
            "mean_nrmse": float(np.mean([m["aggregate"]["mean_nrmse"] for m in metrics_list])),
            "max_nrmse": float(np.max([m["aggregate"]["max_nrmse"] for m in metrics_list])),
        }
        return avg

    def sample_batch(self, batch_size: int) -> dict:
        """
        Sample a batch of states and structural parameters.
        Returns a dict with normalized NN inputs, physical states (k, A),
        physical structural params, and state bounds (k_low, k_high, A_low, A_high).
        """
        p = self.p

        k_norm         = torch.rand(batch_size, device=self.device)
        alpha_norm     = torch.rand(batch_size, device=self.device)
        beta_norm      = torch.rand(batch_size, device=self.device)
        delta_norm     = torch.rand(batch_size, device=self.device)
        rho_norm       = torch.rand(batch_size, device=self.device)
        gamma_norm     = torch.rand(batch_size, device=self.device)
        sigma_eps_norm = torch.rand(batch_size, device=self.device)

        # Oversample hard region: high beta and low delta.
        if p.hard_region_prob > 0.0:
            hard_mask = torch.rand(batch_size, device=self.device) < p.hard_region_prob
            n_hard = int(hard_mask.sum().item())
            if n_hard > 0:
                beta_norm[hard_mask] = p.hard_beta_low_norm + (1.0 - p.hard_beta_low_norm) * torch.rand(
                    n_hard, device=self.device
                )
                delta_norm[hard_mask] = p.hard_delta_high_norm * torch.rand(n_hard, device=self.device)

        A_norm = torch.rand(batch_size, device=self.device)

        # Denormalize structural parameters
        alpha     = self.denormalize(alpha_norm,     p.alpha_bounds[0],     p.alpha_bounds[1])
        beta      = self.denormalize(beta_norm,      p.beta_bounds[0],      p.beta_bounds[1])
        delta     = self.denormalize(delta_norm,     p.delta_bounds[0],     p.delta_bounds[1])
        rho       = self.denormalize(rho_norm,       p.rho_bounds[0],       p.rho_bounds[1])
        gamma     = self.denormalize(gamma_norm,     p.gamma_bounds[0],     p.gamma_bounds[1])
        sigma_eps = self.denormalize(sigma_eps_norm, p.sigma_eps_bounds[0], p.sigma_eps_bounds[1])

        # Compute per-sample state bounds from structural parameters
        A_low, A_high = self._A_bounds_tensors(rho, sigma_eps)
        k_ss, _, _    = self._steady_state_batch(alpha, beta, delta)
        k_low  = p.k_bounds[0] * k_ss
        k_high = p.k_bounds[1] * k_ss

        # Denormalize states to physical space
        k = self.denormalize(k_norm, k_low, k_high)
        A = self.denormalize(A_norm, A_low, A_high)

        # Normalized NN inputs: (k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm)
        inputs = torch.stack(
            [k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm],
            dim=1,
        )
        return dict(
            inputs=inputs,
            k=k, A=A,
            k_low=k_low, k_high=k_high,
            A_low=A_low, A_high=A_high,
            alpha=alpha, beta=beta, delta=delta,
            rho=rho, gamma=gamma, sigma_eps=sigma_eps,
        )

    def compute_residuals(self, batch: dict):
        """Euler equation residuals for a batch produced by sample_batch."""
        inputs    = batch["inputs"]
        k         = batch["k"];         A         = batch["A"]
        k_low     = batch["k_low"];     k_high    = batch["k_high"]
        A_low     = batch["A_low"];     A_high    = batch["A_high"]
        alpha     = batch["alpha"];     beta      = batch["beta"]
        delta     = batch["delta"];     rho       = batch["rho"]
        gamma     = batch["gamma"];     sigma_eps = batch["sigma_eps"]

        # Normalized structural params (columns of inputs) — needed to build inputs_next for NN
        alpha_norm     = inputs[:, 2]; beta_norm      = inputs[:, 3]
        delta_norm     = inputs[:, 4]; rho_norm       = inputs[:, 5]
        gamma_norm     = inputs[:, 6]; sigma_eps_norm = inputs[:, 7]

        # Policy: fraction of resources consumed
        frac = self.model(inputs).squeeze()

        # Current resources (use per-sample alpha, delta)
        resources = A * (k ** alpha) + (1.0 - delta) * k
        c = frac * resources
        # Match simulation dynamics: no state-box clamp, only positivity guards.
        k_next = (resources - c).clamp(min=1e-8)
        c = c.clamp(min=1e-6)

        mu = c ** (-gamma)
        expected_rhs = torch.zeros_like(mu)
        k_next_norm = self.normalize(k_next, k_low, k_high)

        for i in range(self.n_quad):
            eps = self.z_nodes[i]
            weight = self.z_weights[i]
            log_A_next = rho * torch.log(A.clamp(min=1e-8)) + sigma_eps * eps
            A_next = torch.exp(log_A_next)
            A_next_norm = self.normalize(A_next, A_low, A_high)

            inputs_next = torch.stack([
                k_next_norm,
                A_next_norm,
                alpha_norm,
                beta_norm,
                delta_norm,
                rho_norm,
                gamma_norm,
                sigma_eps_norm,
            ], dim=1)

            frac_next = self.model(inputs_next).squeeze()
            resources_next = A_next * (k_next ** alpha) + (1.0 - delta) * k_next
            k_next_next = ((1.0 - frac_next) * resources_next).clamp(min=1e-8)
            c_next = (resources_next - k_next_next).clamp(min=1e-6)
            mu_next = c_next ** (-gamma)
            R_next = alpha * A_next * (k_next ** (alpha - 1.0)) + (1.0 - delta)
            expected_rhs += weight * (beta * mu_next * R_next)

        return expected_rhs - mu

    def train(
        self,
        batch_size: int = 2048,
        epochs: int = 10000,
        eval_every: int = 200,
        val_batch_size: int = 8192,
        patience: int = 20,
        min_rel_improve: float = 5e-3,
        panel_n_cases: int = 4,
        panel_T: int = 120,
        panel_seed: int = 321,
        best_checkpoint_path: str | None = None,
    ):
        """
        Train with early stopping on a fixed validation residual set.
        Also reports fixed-panel NN-vs-TI diagnostics (NRMSE and level ratios).
        """
        self.model.train()
        losses = []
        with torch.no_grad():
            val_batch = self.sample_batch(val_batch_size)
        validation_panel = self._build_validation_panel(panel_n_cases, panel_seed)
        if validation_panel:
            logger.info("Validation panel: %d fixed parameter cases (T=%d)", panel_n_cases, panel_T)

        best_val_loss = np.inf
        best_epoch = 0
        bad_evals = 0
        best_state_dict = None
        logger.info(f"Training over wide parameter range on {self.device}...")
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            
            batch = self.sample_batch(batch_size)
            residuals = self.compute_residuals(batch)
            
            loss = torch.mean(residuals ** 2)
            loss.backward()
            
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % eval_every == 0:
                with torch.no_grad():
                    prev_mode = self.model.training
                    self.model.eval()
                    val_res = self.compute_residuals(val_batch)
                    val_loss = float(torch.mean(val_res ** 2).item())
                    if prev_mode:
                        self.model.train()

                panel_metrics = self._evaluate_validation_panel(validation_panel, panel_T)
                panel_msg = ""
                if panel_metrics:
                    lr = panel_metrics["level_ratio"]
                    panel_msg = (
                        " | panel mean_nrmse={:.3f}, max_nrmse={:.3f},"
                        " level_ratio[c,k,y,i]=[{:.3f},{:.3f},{:.3f},{:.3f}]"
                    ).format(
                        panel_metrics["aggregate"]["mean_nrmse"],
                        panel_metrics["aggregate"]["max_nrmse"],
                        lr["consumption"],
                        lr["capital"],
                        lr["output"],
                        lr["investment"],
                    )
                logger.info(
                    "Epoch %d | train_mse=%.3e | val_mse=%.3e%s",
                    epoch,
                    loss.item(),
                    val_loss,
                    panel_msg,
                )

                rel_improve = (best_val_loss - val_loss) / max(abs(best_val_loss), 1e-12)
                if val_loss < best_val_loss and (np.isinf(best_val_loss) or rel_improve >= min_rel_improve):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    bad_evals = 0
                    best_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    if best_checkpoint_path is not None:
                        self.save(best_checkpoint_path)
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        logger.info(
                            "Early stopping at epoch %d (best val_mse=%.3e at epoch %d).",
                            epoch,
                            best_val_loss,
                            best_epoch,
                        )
                        break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restored best model from epoch %d (val_mse=%.3e).", best_epoch, best_val_loss)
        else:
            logger.warning("No best validation checkpoint captured; keeping last-epoch weights.")
                
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

        _a = torch.tensor(alpha, dtype=torch.float32); _b = torch.tensor(beta, dtype=torch.float32); _d = torch.tensor(delta, dtype=torch.float32)
        k_ss_sim, c_ss_sim, y_ss_sim = (x.item() for x in self._steady_state_batch(_a, _b, _d))
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
        A_low, A_high = self._A_bounds_numpy(rho, sigma_eps)
        k_norm = lambda k: self.normalize(torch.tensor(k, dtype=torch.float32), k_low, k_high)
        A_norm = lambda A: self.normalize(torch.tensor(A, dtype=torch.float32), A_low, A_high)
        alpha_n = self.normalize(torch.tensor(alpha), p.alpha_bounds[0], p.alpha_bounds[1])
        beta_n = self.normalize(torch.tensor(beta), p.beta_bounds[0], p.beta_bounds[1])
        delta_n = self.normalize(torch.tensor(delta), p.delta_bounds[0], p.delta_bounds[1])
        rho_n = self.normalize(torch.tensor(rho), p.rho_bounds[0], p.rho_bounds[1])
        gamma_n = self.normalize(torch.tensor(gamma), p.gamma_bounds[0], p.gamma_bounds[1])
        sigma_eps_n = self.normalize(torch.tensor(sigma_eps), p.sigma_eps_bounds[0], p.sigma_eps_bounds[1])

        with torch.no_grad():
            for t in range(T):
                k, A = k_series[t], A_series[t]
                state = torch.stack([
                    k_norm(k), A_norm(A), alpha_n, beta_n, delta_n, rho_n, gamma_n, sigma_eps_n,
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


def train_rbc_model(
    batch_size: int = 2048,
    epochs: int = 50000,
    eval_every: int = 200,
    val_batch_size: int = 8192,
    patience: int = 20,
    min_rel_improve: float = 5e-3,
    panel_n_cases: int = 4,
    panel_T: int = 120,
    panel_seed: int = 321,
):
    """
    Canonical training entrypoint for RBC NN.
    Saves best/final checkpoint and training loss plot under full-rbc/.
    """
    out_dir = Path(__file__).resolve().parent
    checkpoint_path = out_dir / "rbc_nn.pt"
    loss_plot = out_dir / "learn_rbc_loss.png"

    device = get_device()
    logger.info("Using device: %s", device)
    params = Params()
    solver = RBCSolver(params, device=device)
    losses = solver.train(
        batch_size=batch_size,
        epochs=epochs,
        eval_every=eval_every,
        val_batch_size=val_batch_size,
        patience=patience,
        min_rel_improve=min_rel_improve,
        panel_n_cases=panel_n_cases,
        panel_T=panel_T,
        panel_seed=panel_seed,
        best_checkpoint_path=str(checkpoint_path),
    )
    # Save final model (already restored-to-best by solver.train()).
    solver.save(str(checkpoint_path))
    logger.info("Saved trained checkpoint to %s", checkpoint_path)

    plt.figure(figsize=(6, 4))
    plt.semilogy(losses, alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (Euler residual)")
    plt.title("RBC NN training loss")
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=150)
    plt.close()
    logger.info("Saved %s", loss_plot)
    return solver, losses


def train_rbc_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--val-batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-rel-improve", type=float, default=5e-3)
    parser.add_argument("--panel-n-cases", type=int, default=4)
    parser.add_argument("--panel-T", type=int, default=120)
    parser.add_argument("--panel-seed", type=int, default=321)
    args = parser.parse_args()
    train_rbc_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every=args.eval_every,
        val_batch_size=args.val_batch_size,
        patience=args.patience,
        min_rel_improve=args.min_rel_improve,
        panel_n_cases=args.panel_n_cases,
        panel_T=args.panel_T,
        panel_seed=args.panel_seed,
    )


if __name__ == "__main__":
    train_rbc_cli()