"""
Time Iteration (TI) for the RBC model using cubic splines.
Uses the same Params as learn_rbc so results are comparable at a given calibration.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import RectBivariateSpline

# Use Params from learn_rbc so TI and NN share the same calibration
from learn_rbc import Params

# Set random seeds for reproducibility
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


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
            "investment": i_series,
        }


# When run directly, only solve TI (comparison and plotting live in compare_rbc.py)
if __name__ == "__main__":
    params = Params()
    ti_solver = RBCTISolver(params)
    policy_ti = ti_solver.solve()
    np.random.seed(123)
    results = ti_solver.simulate(policy_ti, T=200)
    logger.info("TI simulation done. Use compare_rbc.py for NN vs TI comparison.")