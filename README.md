# ML-DSGE

Neural-network-based solvers for DSGE models, with an emphasis on economic interpretability and direct benchmarking against classical numerical methods.

Current main implementation: a stochastic RBC model solved by:
- a neural policy approximation (`full-rbc/learn_rbc.py`)
- a Time Iteration benchmark (`full-rbc/rbc_TimeIter.py`)

A Julia port lives in [`full-rbc-julia/`](full-rbc-julia/README.md), with the
same model/diagnostics plus a fix (normalized Euler residual) for divergence
patterns found via `full-rbc/diagnose_divergence.py`.

---

## Motivation

Classical global methods are reliable but expensive and can scale poorly as models gain states, shocks, and nonlinearities.  
This project studies a hybrid workflow:
- train one NN policy over a parameter region,
- validate aggressively against a trusted baseline (TI),
- diagnose where and why discrepancies occur (not just average error).

The goal is research utility: concise code, transparent diagnostics, and reproducible comparison artifacts.

---

## Repository Structure

```text
ML-DSGE/
├── full-rbc/
│   ├── learn_rbc.py             # Canonical training entrypoint + RBC NN solver
│   ├── rbc_TimeIter.py          # Time Iteration (cubic spline) benchmark
│   ├── compare_rbc.py           # Single-calibration NN vs TI comparison
│   ├── diagnose_divergence.py   # Parameter-sweep diagnostics (parallel)
│   ├── simulation/              # Generated divergence outputs (csv/json/png)
│   ├── rbc_nn.pt                # Trained checkpoint (generated)
│   ├── rbc_comparison.png       # Comparison figure (generated)
│   ├── rbc_paths.csv            # Time paths (generated)
│   └── rbc_metrics.json         # Metrics summary (generated)
├── full-rbc-julia/
│   ├── src/                     # RBCParams, TISolver, NNSolver, diagnostics
│   ├── scripts/                 # train.jl, compare.jl, diagnose_divergence.jl
│   └── README.md                # Julia-specific docs incl. the normalization fix
├── poc/
├── lstm/
└── README.md
```

---

## RBC Model (Current Scope)

### States / Control
- State: capital `k`, productivity `A`
- Control: consumption `c`

### Structural parameters
- `alpha`: capital share
- `beta`: discount factor
- `delta`: depreciation rate
- `gamma`: CRRA risk aversion
- `rho`: productivity persistence
- `sigma_eps`: shock innovation std

Productivity law of motion:
- `log A_{t+1} = rho * log A_t + sigma_eps * eps_{t+1}`
- `eps ~ N(0, 1)`

---

## Solver Design

### 1) Neural solver (`learn_rbc.py`)

Policy network predicts consumption share from normalized inputs:
- Inputs (8D): `(k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm)`
- Output: `frac in (0,1)` via sigmoid
- Consumption: `c = frac * resources`

Training objective:
- Euler equation residual MSE with Hermite-Gauss quadrature expectation.

Important implementation details:
- Dynamic productivity normalization uses stationary log-TFP scale:
  - `sigma_stat = sigma_eps / sqrt(1 - rho^2)`
  - `A` support roughly `exp(± A_sigma_mult * sigma_stat)`.
- Training uses early stopping on a fixed validation residual set.
- Fixed TI validation panel is pre-built and evaluated during training.

### 2) Time Iteration benchmark (`rbc_TimeIter.py`)

- Cubic-spline policy over (`k`, `A`) grid.
- Same calibration object (`Params`) so NN/TI are directly comparable.
- Same shock seed is used during comparisons.

---

## End-to-End Workflow

### Train NN checkpoint

```bash
python3 full-rbc/learn_rbc.py
```

This writes:
- `full-rbc/rbc_nn.pt`
- `full-rbc/learn_rbc_loss.png`

### Compare one calibration (NN vs TI)

```bash
python3 full-rbc/compare_rbc.py
```

This writes:
- `full-rbc/rbc_comparison.png`
- `full-rbc/rbc_paths.csv`
- `full-rbc/rbc_metrics.json`

### Sweep many calibrations and diagnose divergence

```bash
python3 full-rbc/diagnose_divergence.py --n-cases 40 --top-k 5 --T 200 --n-workers 8
```

Outputs under `full-rbc/simulation/`:
- `divergence_summary.csv`
- `divergence_top_cases.json`
- `divergence_case_XXX_paths.csv`
- `divergence_case_XXX_plot.png`

---

## Interpreting Diagnostics

### Core metrics
- `rmse`: absolute path error
- `nrmse_vs_ti_std = rmse / std(TI series)`: scale-free error
- `mean_nrmse`, `max_nrmse`: aggregate diagnostics
- `nn_k_oob_frac`, `ti_k_oob_frac`: share of periods outside solver support

### Practical reading guide
- `nrmse_A = 0` and overlapping TFP lines mean shock process alignment is correct.
- Large `nrmse_k` with high correlation can indicate mostly level bias.
- Large `nrmse_k` with low/negative correlation indicates dynamic mismatch.
- High OOB fractions usually signal support mismatch rather than pure policy error.

---

## Current Lessons from Experiments

- Matching training dynamics and simulation dynamics matters a lot.
  - When training residual transitions were overly clamped but simulation was not, divergence worsened.
- Aggressive targeted oversampling (high-`beta`, low-`delta`) can introduce systematic level bias.
- Wider/deeper nets alone do not guarantee better fit if objective/sampling are misaligned.
- Diagnostics that include path-level artifacts are essential to avoid being misled by scalar loss alone.

---

## Configuration Notes

Main parameter ranges are defined in `Params` inside `learn_rbc.py`.  
These include:
- structural bounds (`alpha_bounds`, `beta_bounds`, `delta_bounds`, `rho_bounds`, `gamma_bounds`, `sigma_eps_bounds`)
- state scaling controls (`k_bounds`, `A_sigma_mult`)
- training sampling controls (including optional hard-region oversampling parameters)

For reproducible experiments, change one mechanism at a time and compare `simulation/divergence_summary.csv` across runs.

---

## Dependencies

- Python 3.10+ (recommended)
- `torch`
- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`

All dependencies are pinned in `requirements.txt`. Install them into your environment (e.g. a conda env) with:

```bash
python3 -m pip install -r requirements.txt
```

If `ModuleNotFoundError: No module named 'torch'` appears, install `torch` into the same interpreter used to run scripts:

```bash
python3 -m pip show torch
python3 -m pip install torch
```

---

## Roadmap

- [ ] Check why divergence occur in full-rbc
- [ ] [ ] Extend the same workflow to richer DSGE settings (e.g., NK, heterogeneous agents).
- [ ] Introduce rollout-aware training terms for long-horizon dynamic consistency.
- [ ] Improve ranking diagnostics with blended scores (normalized + level + correlation)
- [ ] Standardize experiment tracking across model/seed/config runs.

---

## License

MIT
