import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for SciPy
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Define dummy class to satisfy linter/static analysis
    class RectBivariateSpline:
        def __init__(self, *args, **kwargs): pass
        def ev(self, x, y): return np.array([0.0])
    logger.warning("SciPy not found. Time Iteration comparison will be skipped.")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class Params:
    """Model parameters for the RBC model."""
    alpha: float = 0.30      # capital share
    beta: float = 0.95       # discount factor
    delta: float = 0.1       # depreciation rate
    gamma: float = 2.0       # risk aversion
    rho: float = 0.90        # persistence of productivity shock
    sigma_eps: float = 0.02  # std dev of shock innovation

    # Bounds for state space sampling
    k_bounds: tuple = (0.5, 1.5) # fraction of SS capital
    A_bounds: tuple = (0.5, 1.5)
    beta_bounds: tuple = (0.9, 0.99)
    rho_bounds: tuple = (0.9, 0.99)
    gamma_bounds: tuple = (0.5, 4.0)

class RBCNet(nn.Module):
    """Neural Network to approximate the policy function."""
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, output_bias: float = 0.0):
        super(RBCNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            #nn.init.xavier_uniform_(layers[-2].weight, gain= 0.01)
            prev_dim = h_dim
        
        # Output layer
        output_layer = nn.Linear(prev_dim, output_dim)
        #nn.init.xavier_uniform_(output_layer.weight, gain= 0.01)
        # Initialize bias to start close to steady state
        with torch.no_grad():
            output_layer.bias.fill_(output_bias)
        
        layers.append(output_layer)
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RBCSolver:
    def __init__(self, params: Params, device: str = "cpu"):
        self.p = params
        self.device = torch.device(device)
        
        # Calculate steady state values
        self.k_ss, self.c_ss, self.y_ss, self.A_ss = self.steady_state()
        logger.info(f"Steady-state capital (A=1): {self.k_ss:.3f}")

        # Initialize Neural Network
        # Calculate initial bias
        res_ss_init = self.y_ss + (1.0 - self.p.delta) * self.k_ss
        frac_ss_init = self.c_ss / res_ss_init
        # Inverse sigmoid: log(y / (1-y))
        init_bias = np.log(frac_ss_init / (1.0 - frac_ss_init))
        logger.info(f"Initializing output bias to: {init_bias:.3f} (SS frac: {frac_ss_init:.3f})")

        # 5 inputs: k, A, beta, rho, gamma
        self.model = RBCNet(5, [128, 64, 32], 1, output_bias=init_bias).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        
        # Quadrature nodes (Hermite-Gauss)
        # 5 nodes
        self.n_quad = 3
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        # Transform from e^(-x^2) to N(0,1)
        # nodes = sqrt(2) * x_i
        # weights = w_i / sqrt(pi)
        self.z_nodes = torch.tensor(nodes * np.sqrt(2), dtype=torch.float32, device=self.device)
        self.z_weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32, device=self.device)

    def steady_state(self):
        """Computes steady state values for A=1."""
        alpha, beta, delta = self.p.alpha, self.p.beta, self.p.delta
        A_ss = 1.0
        # Euler: 1 = beta * (alpha * A * k^(alpha-1) + 1 - delta)
        # k^(alpha-1) = (1/beta - (1-delta)) / (alpha * A)
        term = (1.0 / beta - (1.0 - delta)) / (alpha * A_ss)
        k_ss = term ** (1.0 / (alpha - 1.0))
        y_ss = A_ss * k_ss ** alpha
        c_ss = y_ss - delta * k_ss
        return k_ss, c_ss, y_ss, A_ss

    def normalize(self, x, x_low, x_high):
        return (x - x_low) / (x_high - x_low)

    def denormalize(self, x, x_low, x_high):
        return x * (x_high - x_low) + x_low

    def sample_batch(self, batch_size: int):
        """Generates a batch of state variables."""
        p = self.p

        # Uniform sampling for k, beta, rho, gamma
        k_batch = torch.rand(batch_size, device=self.device)
        beta_batch = torch.rand(batch_size, device=self.device)
        rho_batch = torch.rand(batch_size, device=self.device)
        gamma_batch = torch.rand(batch_size, device=self.device)

        # Lognormal sampling for A
        # Stationary std dev: sigma_stat = sigma_eps / sqrt(1 - rho^2)
        # Note: rho here is a random variable, so we compute sigma_stat per sample
        rho_val = self.denormalize(rho_batch, p.rho_bounds[0], p.rho_bounds[1])
        sigma_stat = p.sigma_eps / torch.sqrt(1.0 - rho_val**2)

        A = torch.exp(sigma_stat * torch.randn(batch_size, device=self.device))
        A_batch = self.normalize(A, p.A_bounds[0], p.A_bounds[1])

        # Stack inputs: shape (batch_size, 5)
        # Order: k, A, beta, rho, gamma
        inputs = torch.stack([k_batch, A_batch, beta_batch, rho_batch, gamma_batch], dim=1)

        return inputs

    def compute_residuals(self, inputs: torch.Tensor):
        """Computes Euler residuals for a batch of inputs."""
        p = self.p

        # Unpack normalized inputs
        k_norm = inputs[:, 0]
        A_norm = inputs[:, 1]
        beta_norm = inputs[:, 2]
        rho_norm = inputs[:, 3]
        gamma_norm = inputs[:, 4]

        # Denormalize parameters needed for computation
        beta = self.denormalize(beta_norm, p.beta_bounds[0], p.beta_bounds[1])
        rho = self.denormalize(rho_norm, p.rho_bounds[0], p.rho_bounds[1])
        gamma = self.denormalize(gamma_norm, p.gamma_bounds[0], p.gamma_bounds[1])

        # Denormalize states
        k_low = p.k_bounds[0] * self.k_ss
        k_high = p.k_bounds[1] * self.k_ss
        k = self.denormalize(k_norm, k_low, k_high)
        A = self.denormalize(A_norm, p.A_bounds[0], p.A_bounds[1])

        # Get policy (fraction of resources consumed)
        frac = self.model(inputs).squeeze() # shape (batch_size,)

        # Current resources
        resources = A * (k ** p.alpha) + (1.0 - p.delta) * k
        c = frac * resources

        # Next capital
        k_next = resources - c
        # Clamp k_next to stay within bounds for stability (and logic)
        k_next = torch.clamp(k_next, k_low, k_high)

        # Re-calculate c consistent with clamped k_next (budget constraint)
        c = resources - k_next
        c = torch.clamp(c, min=1e-6) # Numerical safety

        # Marginal utility at t
        mu = c ** (-gamma)

        # Compute Expectation using Quadrature
        # We need to compute RHS for each quadrature node
        expected_rhs = torch.zeros_like(mu)

        k_next_norm = self.normalize(k_next, k_low, k_high)

        for i in range(self.n_quad):
            eps = self.z_nodes[i]
            weight = self.z_weights[i]

            # Next A
            # log(A') = rho * log(A) + sigma_eps * eps
            log_A_next = rho * torch.log(A) + p.sigma_eps * eps
            A_next = torch.exp(log_A_next)

            # Normalize A_next for network input
            A_next_norm = self.normalize(A_next, p.A_bounds[0], p.A_bounds[1])

            # Prepare next state inputs
            # Only k and A change; beta, rho, gamma stay same (parameters)
            inputs_next = torch.stack([
                k_next_norm,
                A_next_norm,
                beta_norm,
                rho_norm,
                gamma_norm
            ], dim=1)

            # Predict policy at t+1
            frac_next = self.model(inputs_next).squeeze()

            # Resources at t+1
            resources_next = A_next * (k_next ** p.alpha) + (1.0 - p.delta) * k_next

            # k''
            k_next_next = (1.0 - frac_next) * resources_next
            k_next_next = torch.clamp(k_next_next, k_low, k_high)

            # c at t+1
            c_next = resources_next - k_next_next
            c_next = torch.clamp(c_next, min=1e-6)

            # Marginal utility at t+1
            mu_next = c_next ** (-gamma)

            # Return at t+1
            R_next = p.alpha * A_next * (k_next ** (p.alpha - 1.0)) + (1.0 - p.delta)

            # Accumulate weighted expectation
            expected_rhs += weight * (beta * mu_next * R_next)

        # Residual
        residuals = expected_rhs - mu
        return residuals

    def train(self, batch_size: int = 2048, epochs: int = 10000):
        self.model.train()
        losses = []
        
        logger.info(f"Starting training on {self.device}...")
        
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

    def simulate(self, T: int = 200, k0: float = None, A0: float = None) -> dict:
        """Simulate the economy."""
        self.model.eval()
        p = self.p
        
        if k0 is None: k0 = self.k_ss
        if A0 is None: A0 = self.A_ss
        
        # Prepare arrays
        k_series = np.zeros(T + 1)
        A_series = np.zeros(T + 1)
        c_series = np.zeros(T)
        y_series = np.zeros(T)
        i_series = np.zeros(T)
        
        k_series[0] = k0
        A_series[0] = A0
        
        # Shocks
        eps_series = np.random.randn(T)
        
        # Constant parameters for simulation (using mean values)
        beta_val = p.beta
        rho_val = p.rho
        gamma_val = p.gamma
        
        # Normalize constant params
        beta_norm = self.normalize(torch.tensor(beta_val), p.beta_bounds[0], p.beta_bounds[1])
        rho_norm = self.normalize(torch.tensor(rho_val), p.rho_bounds[0], p.rho_bounds[1])
        gamma_norm = self.normalize(torch.tensor(gamma_val), p.gamma_bounds[0], p.gamma_bounds[1])
        
        k_low = p.k_bounds[0] * self.k_ss
        k_high = p.k_bounds[1] * self.k_ss

        with torch.no_grad():
            for t in range(T):
                k = k_series[t]
                A = A_series[t]
                
                # Normalize state
                k_norm = self.normalize(torch.tensor(k, dtype=torch.float32), k_low, k_high)
                A_norm = self.normalize(torch.tensor(A, dtype=torch.float32), p.A_bounds[0], p.A_bounds[1])
                
                # Create input (1, 5)
                state = torch.stack([k_norm, A_norm, beta_norm, rho_norm, gamma_norm]).unsqueeze(0).to(self.device)
                
                # Get policy
                frac = self.model(state).item()
                
                # Economy dynamics
                y = A * k ** p.alpha
                y_series[t] = y
                
                resources = y + (1.0 - p.delta) * k
                c = frac * resources
                c_series[t] = c
                
                inv = y - c
                i_series[t] = inv
                
                k_next = (1.0 - p.delta) * k + inv
                k_series[t+1] = max(k_next, 1e-6)
                
                # A dynamics
                log_A_next = rho_val * np.log(max(A, 1e-8)) + p.sigma_eps * eps_series[t]
                A_series[t+1] = np.exp(log_A_next)
                
        return {
            "capital": k_series[:T],
            "productivity": A_series[:T],
            "consumption": c_series,
            "output": y_series,
            "investment": i_series
        }

class RBCTISolver:
    """
    Solves the RBC model using Time Iteration with Cubic Splines.
    """
    def __init__(self, params: Params):
        self.p = params
        self.k_ss, self.c_ss, self.y_ss, self.A_ss = self.calculate_steady_state()
        
        # Grids
        self.n_k = 30
        self.n_A = 15
        
        self.k_min = self.p.k_bounds[0] * self.k_ss
        self.k_max = self.p.k_bounds[1] * self.k_ss
        self.A_min = self.p.A_bounds[0] * self.A_ss
        self.A_max = self.p.A_bounds[1] * self.A_ss
        
        self.k_nodes = np.linspace(self.k_min, self.k_max, self.n_k)
        self.A_nodes = np.linspace(self.A_min, self.A_max, self.n_A)
        
        # Initial guess: c = c_ss
        self.c_policy = np.full((self.n_k, self.n_A), self.c_ss)
        
        # Quadrature
        self.n_quad = 7
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        self.z_nodes = nodes * np.sqrt(2)
        self.z_weights = weights / np.sqrt(np.pi)

    def calculate_steady_state(self):
        alpha, beta, delta = self.p.alpha, self.p.beta, self.p.delta
        A_ss = 1.0
        term = (1.0 / beta - (1.0 - delta)) / (alpha * A_ss)
        k_ss = term ** (1.0 / (alpha - 1.0))
        y_ss = A_ss * k_ss ** alpha
        c_ss = y_ss - delta * k_ss
        return k_ss, c_ss, y_ss, A_ss

    def solve(self, tol=1e-6, max_iter=1000, damping=0.5):
        logger.info("Starting Time Iteration (SciPy Splines)...")
        
        for iter_idx in range(max_iter):
            c_new = np.empty_like(self.c_policy)
            
            # Current policy interpolator
            interp = RectBivariateSpline(self.k_nodes, self.A_nodes, self.c_policy, kx=3, ky=3)
            
            # Loop over nodes
            for i in range(self.n_k):
                for j in range(self.n_A):
                    k = self.k_nodes[i]
                    A = self.A_nodes[j]
                    
                    c_old = self.c_policy[i, j]
                    c_old = max(c_old, 1e-6)
                    
                    y = A * k ** self.p.alpha
                    res = y + (1 - self.p.delta) * k
                    
                    c_calc = min(c_old, res - 1e-4)
                    k_prime = res - c_calc
                    
                    # Clamp k_prime
                    k_prime_clamped = np.clip(k_prime, self.k_min, self.k_max)
                    
                    E_rhs = 0.0
                    for z in range(self.n_quad):
                        eps = self.z_nodes[z]
                        logA_prime = self.p.rho * np.log(A) + self.p.sigma_eps * eps
                        A_prime = np.exp(logA_prime)
                        
                        # Clamp A_prime
                        A_prime_clamped = np.clip(A_prime, self.A_min, self.A_max)
                        
                        c_prime = interp.ev(k_prime_clamped, A_prime_clamped).item()
                        c_prime = max(c_prime, 1e-6)
                        
                        mu_prime = c_prime ** (-self.p.gamma)
                        R_prime = self.p.alpha * A_prime * k_prime_clamped ** (self.p.alpha - 1) + (1 - self.p.delta)
                        
                        E_rhs += self.z_weights[z] * (mu_prime * R_prime)
                    
                    c_target = (self.p.beta * E_rhs) ** (-1 / self.p.gamma)
                    c_new[i, j] = damping * c_target + (1 - damping) * c_old
            
            diff = np.max(np.abs(c_new - self.c_policy))
            if iter_idx % 10 == 0:
                logger.info(f"TI Iteration {iter_idx}, diff: {diff:.6f}")
            
            if diff < tol:
                logger.info(f"TI Converged in {iter_idx} iterations")
                self.c_policy = c_new
                return RectBivariateSpline(self.k_nodes, self.A_nodes, self.c_policy, kx=3, ky=3)
            
            self.c_policy = c_new
            
        logger.warning("TI Did not converge")
        return RectBivariateSpline(self.k_nodes, self.A_nodes, self.c_policy, kx=3, ky=3)

    def simulate(self, policy_interp, T=200, k0=None, A0=None):
        if k0 is None: k0 = self.k_ss
        if A0 is None: A0 = self.A_ss
        
        k_series = np.zeros(T + 1)
        A_series = np.zeros(T + 1)
        c_series = np.zeros(T)
        y_series = np.zeros(T)
        i_series = np.zeros(T)
        
        k_series[0] = k0
        A_series[0] = A0
        
        eps_series = np.random.randn(T)
        
        for t in range(T):
            k = k_series[t]
            A = A_series[t]
            
            # Check bounds for potential extrapolation issues
            if k < self.k_min or k > self.k_max or A < self.A_min or A > self.A_max:
                # We do not clamp here to match Julia's behavior (which likely extrapolates),
                # but we note that this is a source of potential divergence.
                pass

            c = policy_interp.ev(k, A).item()
            c_series[t] = c
            
            y = A * k ** self.p.alpha
            y_series[t] = y
            
            i = y - c
            i_series[t] = i
            
            k_next = y + (1 - self.p.delta) * k - c
            k_series[t+1] = k_next
            
            logA_next = self.p.rho * np.log(max(A, 1e-8)) + self.p.sigma_eps * eps_series[t]
            A_series[t+1] = np.exp(logA_next)
            
        return {
            "capital": k_series[:T],
            "productivity": A_series[:T],
            "consumption": c_series,
            "output": y_series,
            "investment": i_series
        }

def run_impulse_response_logic(solver: RBCSolver, shock_size=0.05, T=40, NIRF=100):
    # Accumulators for levels
    # We need to accumulate output, consumption, investment, capital
    
    def get_data(res):
        return np.stack([res["output"], res["consumption"], res["investment"], res["capital"]], axis=1)
        
    base_sum = np.zeros((T, 4))
    shock_sum = np.zeros((T, 4))
    
    for i in range(NIRF):
        seed = 9000 + i
        
        # Baseline
        np.random.seed(seed)
        base = solver.simulate(T, solver.k_ss, solver.A_ss)
        base_sum += get_data(base)
        
        # Shocked
        np.random.seed(seed)
        shock = solver.simulate(T, solver.k_ss, solver.A_ss * (1 + shock_size))
        shock_sum += get_data(shock)
        
    base_avg = base_sum / NIRF
    shock_avg = shock_sum / NIRF
    
    # Percent deviation
    # Avoid division by zero if any (unlikely near SS)
    avg_dev = 100 * (shock_avg / base_avg - 1.0)
    
    # Plot
    labels = ["Output", "Consumption", "Investment", "Capital"]
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(labels):
        plt.plot(avg_dev[:, idx], label=label, linewidth=2)
        
    plt.title(f"Impulse Responses to {shock_size*100:.1f}% Productivity Shock")
    plt.xlabel("Time")
    plt.ylabel("% Deviation from Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("NN_Impulse_Response_Py.png")
    logger.info("Impulse response figure saved.")

def run_impulse_response_ti(solver: RBCTISolver, policy_interp, shock_size=0.05, T=40, NIRF=100):
    # Accumulators for levels
    
    def get_data(res):
        return np.stack([res["output"], res["consumption"], res["investment"], res["capital"]], axis=1)
        
    base_sum = np.zeros((T, 4))
    shock_sum = np.zeros((T, 4))
    
    for i in range(NIRF):
        seed = 9000 + i
        
        # Baseline
        np.random.seed(seed)
        base = solver.simulate(policy_interp, T, solver.k_ss, solver.A_ss)
        base_sum += get_data(base)
        
        # Shocked
        np.random.seed(seed)
        shock = solver.simulate(policy_interp, T, solver.k_ss, solver.A_ss * (1 + shock_size))
        shock_sum += get_data(shock)
        
    base_avg = base_sum / NIRF
    shock_avg = shock_sum / NIRF
    
    # Percent deviation
    avg_dev = 100 * (shock_avg / base_avg - 1.0)
    
    # Plot
    labels = ["Output", "Consumption", "Investment", "Capital"]
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(labels):
        plt.plot(avg_dev[:, idx], label=label, linewidth=2)
        
    plt.title(f"TI Impulse Responses to {shock_size*100:.1f}% Productivity Shock")
    plt.xlabel("Time")
    plt.ylabel("% Deviation from Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("TI_Impulse_Response_Py.png")
    logger.info("TI Impulse response figure saved.")

def check_potential_problems(params: Params):
    """
    Checks for potential sources of discrepancy between NN and TI.
    """
    logger.info("Checking for potential discrepancies...")

    # 1. Grid Bounds vs Distribution
    # TI is bounded by k_bounds and A_bounds. NN is trained on samples.
    # If simulation goes outside bounds, TI clamps/extrapolates, NN extrapolates.
    # A is lognormal, so it has infinite support.
    sigma_stat = params.sigma_eps / np.sqrt(1 - params.rho**2)
    log_A_std = sigma_stat
    # 99% interval for A
    A_low_99 = np.exp(-2.58 * log_A_std)
    A_high_99 = np.exp(2.58 * log_A_std)

    if A_low_99 < params.A_bounds[0] or A_high_99 > params.A_bounds[1]:
        logger.warning(f"Potential Issue: A distribution (99% interval [{A_low_99:.3f}, {A_high_99:.3f}]) "
                       f"exceeds grid bounds [{params.A_bounds[0]}, {params.A_bounds[1]}]. "
                       "TI relies on extrapolation here, which may diverge from NN.")

    # 2. Interpolation
    logger.info("Note: TI uses Cubic Splines (local), NN uses Global approximation. "
                "Differences in smoothness and extrapolation behavior are expected.")

    # 3. Simulation Clamping
    logger.info("Note: NN simulation clamps capital to be positive (max(k, 1e-6)). "
                "TI simulation (in Julia and here) does not explicitly clamp k_next. "
                "If TI policy leads to negative capital (e.g. due to extrapolation error), "
                "the simulation will crash or produce NaNs, whereas NN will survive.")

def main():
    params = Params()
    # Detect MPS (Apple Silicon) or CUDA
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    logger.info(f"Using device: {device}")
    
    solver = RBCSolver(params, device=device)
    
    # Train
    losses = solver.train(batch_size=2048, epochs=5000) 
    
    # Plot Loss
    plt.figure()
    plt.plot(losses[100:])
    plt.title("Training Loss (MSE Euler Residual)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.savefig("training_loss_py.png")
    
    # Impulse Response
    run_impulse_response_logic(solver)
    
    # ==========================================
    # COMPARISON: Time Iteration vs Neural Network
    # ==========================================
    
    check_potential_problems(params)
    
    if SCIPY_AVAILABLE:
        # Time Iteration
        ti_solver = RBCTISolver(params)
        policy_ti = ti_solver.solve()
        
        # Comparison Simulation
        T_sim = 200
        
        # Set seed for NN simulation
        np.random.seed(123)
        nn_results = solver.simulate(T=T_sim)
        
        # Set SAME seed for TI simulation
        np.random.seed(123)
        ti_results = ti_solver.simulate(policy_ti, T=T_sim)
        
        # Plot Comparison
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        axes[0].plot(nn_results["consumption"], label="NN Consumption", linewidth=2)
        axes[0].plot(ti_results["consumption"], label="TI Consumption", linewidth=2, linestyle="--")
        axes[0].legend()
        axes[0].set_title("Consumption Comparison")
        
        axes[1].plot(nn_results["capital"], label="NN Capital", linewidth=2)
        axes[1].plot(ti_results["capital"], label="TI Capital", linewidth=2, linestyle="--")
        axes[1].legend()
        axes[1].set_title("Capital Comparison")
        
        plt.tight_layout()
        plt.savefig("TI_vs_NN_Py.png")
        logger.info("Comparison figure saved.")
        
        # TI Impulse Response
        run_impulse_response_ti(ti_solver, policy_ti)

    else:
        logger.info("Skipping TI comparison due to missing SciPy.")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
