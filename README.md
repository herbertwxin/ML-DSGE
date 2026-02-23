# ML-DSGE: Solving DSGE Models with Machine Learning

This repository implements machine learning techniques to solve Dynamic Stochastic General Equilibrium (DSGE) models, with a focus on neural network-based policy function approximation.

## Overview

Traditional methods for solving DSGE models (e.g., perturbation, projection, value function iteration) can be computationally expensive and may struggle with high-dimensional state spaces or non-linearities. This project explores **neural networks as universal function approximators** to learn policy functions that satisfy the model's equilibrium conditions (e.g., Euler equations) across a wide range of parameter values.

## General Approach

### Core Methodology

1. **Policy Function Approximation**: Instead of solving for the exact policy function analytically, we train a neural network to approximate it by minimizing Euler equation residuals.

2. **Wide Parameter Space Learning**: Rather than solving for a single calibration, the neural network learns the policy function over a **range of structural parameters** (e.g., discount factor, risk aversion, capital share). This allows the model to generalize across different calibrations without retraining.

3. **Residual-Based Training**: The network is trained to satisfy the model's equilibrium conditions (typically Euler equations) by minimizing the squared residuals:
   ```
   Loss = E[(Euler_residual)²]
   ```
   where the expectation is taken over random samples of state variables and parameters.

4. **Comparison with Traditional Methods**: Results are validated against traditional solution methods (e.g., Time Iteration with cubic splines) to ensure accuracy.

### Key Advantages

- **Scalability**: Neural networks can handle high-dimensional state spaces more efficiently than grid-based methods.
- **Generalization**: A single trained network can approximate policies across a wide parameter space.
- **Flexibility**: Easy to extend to more complex models (e.g., multiple agents, additional shocks, non-linearities).
- **Differentiability**: Automatic differentiation enables efficient gradient-based optimization.

## Project Structure

```
ML-DSGE/
├── full-rbc/          # Full RBC model implementation
│   ├── learn_rbc.py      # Neural network solver for RBC
│   ├── rbc_TimeIter.py    # Traditional Time Iteration solver
│   └── compare_rbc.py     # Comparison script (NN vs TI)
├── poc/               # Proof-of-concept implementations
│   └── neural_net.py      # Early neural network experiments
├── lstm/              # LSTM-based approaches (exploratory)
└── README.md          # This file
```

## Current Implementation: Real Business Cycle (RBC) Model

### Model Description

The RBC model features:
- **State variables**: Capital stock (k) and productivity (A)
- **Control variable**: Consumption (c)
- **Structural parameters**: Capital share (α), discount factor (β), depreciation (δ), risk aversion (γ), persistence (ρ), shock volatility (σ_ε)

### Neural Network Architecture

- **Inputs**: Normalized (k, A, α, β, δ, ρ, γ) — 7 dimensions
- **Output**: Consumption fraction (sigmoid → [0,1])
- **Architecture**: Feedforward network with ELU activations
- **Training**: Adam optimizer minimizing Euler equation residuals

### Usage

1. **Train the neural network** (or load from checkpoint):
   ```bash
   python full-rbc/learn_rbc.py
   ```
   This trains over a wide parameter range and saves the model to `rbc_nn.pt`.

2. **Compare with Time Iteration**:
   ```bash
   python full-rbc/compare_rbc.py
   ```
   This loads the trained NN (or trains if missing), solves via TI, runs simulations with the same seed, and generates comparison plots.

3. **Specify calibration**: Edit `get_calibration_params()` in `compare_rbc.py` to change the parameter set used for comparison.

## Technical Details

### Training Process

- **Sampling**: Random batches of (k, A, α, β, δ, ρ, γ) are sampled uniformly within specified bounds.
- **Residual Computation**: For each sample, the Euler equation residual is computed using Hermite-Gauss quadrature for expectations.
- **Steady State**: Per-sample steady-state capital is computed to maintain consistent state-space scaling across parameters.

### Validation

- **Euler Residuals**: Training loss measures how well the network satisfies equilibrium conditions.
- **Simulation Comparison**: Side-by-side comparison with Time Iteration at the same calibration and shock sequence.
- **Parameter Generalization**: Test the network at parameter values not seen during training.

## Future Directions

- Extend to more complex DSGE models (e.g., New Keynesian, heterogeneous agents)
- Explore alternative architectures (e.g., attention mechanisms, graph neural networks)
- Investigate uncertainty quantification and robustness
- Compare with other ML approaches (e.g., reinforcement learning, physics-informed neural networks)

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- SciPy (for Time Iteration comparison)

## References

This work is inspired by recent research on using neural networks for solving economic models, including:
- Deep learning for solving high-dimensional PDEs (which share structure with HJB equations)
- Universal function approximation for policy functions
- Residual-based training for economic equilibrium conditions

## License

[Specify your license here]
