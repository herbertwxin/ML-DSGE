# ML-DSGE

Neural-network-based solvers for DSGE models, with an emphasis on economic
interpretability and direct benchmarking against classical numerical methods.

The main implementation lives in [`rbc/`](rbc/) (Julia): a stochastic RBC
model solved two ways and compared head-to-head:

- **Neural solver** (`rbc/src/engine/` + `rbc/src/model.jl`) — a policy
  network trained on the normalized Euler-equation residual over a whole box
  of structural parameters.
- **Time Iteration benchmark** (`rbc/src/time_iteration.jl`) — Coleman time
  iteration (exact per-node Euler solve + cubic-spline policy) at a single
  calibration. Solved using bisection.

The original Python implementation it was ported from is kept in
[`archive/full-rbc/`](archive/full-rbc/); see
[Archived Python implementation](#archived-python-implementation).

---

## Motivation

Classical global methods are reliable but expensive and can scale poorly as models gain states, shocks, and nonlinearities.  
This project studies a hybrid workflow:
- train one NN policy over a parameter region,
- validate aggressively against a trusted baseline (TI),
- diagnose where and why discrepancies occur (not just average error).

The goal is research utility: concise code, transparent diagnostics, and reproducible comparison artifacts.

---

## Repository structure

The Julia source is split into a **model-agnostic engine** and the **RBC
model**. Nothing under `rbc/src/engine/` references the RBC economics; the
engine reaches the model only through the interface functions in
`rbc/src/engine/interface.jl`, dispatched on the parameter-struct type.

```text
ML-DSGE/
├── README.md                        # this file
├── rbc/                             # main implementation (Julia)
│   ├── Project.toml / Manifest.toml # pinned dependencies
│   ├── src/
│   │   ├── FullRBC.jl               # module entry point
│   │   ├── engine/                  # model-agnostic (never edit to add a model)
│   │   │   ├── interface.jl         # the model interface: hooks a model implements
│   │   │   ├── device.jl            # CPU/GPU selection, default batch sizes
│   │   │   ├── quadrature.jl        # Gauss-Hermite quadrature
│   │   │   ├── network.jl           # policy-net builder + NNSolver container
│   │   │   ├── training.jl          # train! loop, early stopping, checkpoints
│   │   │   └── diagnostics.jl       # solver-vs-benchmark gap metrics
│   │   ├── model.jl                 # THE RBC MODEL: params, primitives, steady state,
│   │   │                            #   sample_batch, Euler residuals + training loss,
│   │   │                            #   NNPolicy, NN-vs-TI validation panel
│   │   ├── time_iteration.jl        # TISolver/TIPolicy: Coleman time iteration
│   │   └── simulate.jl              # one simulate() shared by every policy type
│   ├── scripts/
│   │   ├── common.jl                # shared CLI/IO/Makie-figure helpers
│   │   ├── train.jl                 # train the NN checkpoint
│   │   ├── compare.jl               # single-calibration NN vs TI comparison
│   │   └── diagnose_divergence.jl   # parameter-sweep divergence diagnostics
│   ├── main.qmd                     # Quarto note: simulate the trained NN vs TI
│   ├── simulate.jl                  # standalone simulation script (self-contained)
│   ├── simulation/                  # generated divergence outputs (csv/json/png)
│   ├── rbc_nn.bson                  # trained checkpoint (generated)
│   ├── rbc_comparison.png           # comparison figure (generated)
│   ├── rbc_paths.csv                # time paths (generated)
│   └── rbc_metrics.json             # metrics summary (generated)
├── archive/                         # earlier implementations and experiments
│   └── full-rbc/                    # original Python version (learn_rbc.py, rbc_TimeIter.py, ...)
├── comment/                         # working notes
└── requirements.txt                 # Python deps (archive only)
```

### Adding a model / swapping the loss

A model is a parameter struct (structural scalars + training bounds, like
`RBCParams`) plus methods on that type for the hooks in
`rbc/src/engine/interface.jl`:

| hook | role |
|---|---|
| `policy_spec(p)` | network architecture: input/output dims, hidden widths, output bias |
| `sample_batch(p, rng, n)` | training batch over the normalized state × parameter box |
| `training_loss(p, model, batch, quad; kw...)` | the objective — residuals + penalties |
| `validation_report(p, model, batch, quad; kw...)` | loss components logged at eval points (optional) |
| `build_validation_panel(p, n, seed)` / `evaluate_validation_panel(p, solver, panel, T)` | benchmark diagnostics during training (optional) |

Loss hyperparameters (e.g. `k_oob_weight`) travel through `train!` as an
opaque `loss_kwargs` named tuple, so changing the loss — reweighting terms,
adding penalties, new equilibrium conditions — only ever edits `model.jl`.

---

## Model

- State: capital `k`, productivity `A`; control: consumption `c`.
- Preferences: CRRA, `u'(c) = c^(-gamma)`.
- Technology: `y = A k^alpha`, depreciation `delta`.
- Productivity: `log A' = rho log A + sigma_eps * eps`, `eps ~ N(0,1)`.
- Structural parameters `(alpha, beta, delta, gamma, rho, sigma_eps)` live in
  `RBCParams` (`rbc/src/model.jl`) together with the training bounds and
  state-space controls (`k_bounds`, `A_sigma_mult`).

Both solvers represent the policy as a **consumption share**
`frac = c / resources ∈ (0, 1)` rather than a consumption level. A share is
self-correcting when the simulated state leaves the fitted box (apply the
boundary share to the true resources), whereas extrapolating a raw
consumption level can violate the budget constraint and explode — this bit us
in earlier versions of the TI benchmark.

## Design: one policy interface, one simulate

Anything that implements `consumption_share(policy, k, A)` is a policy:

- `TIPolicy` — cubic B-spline (Interpolations.jl) of the converged share on
  the TI grid, query points clamped to the grid box;
- `NNPolicy` — the trained network evaluated at one fixed calibration
  (normalized parameter inputs precomputed).

A single `simulate(policy, p::RBCParams; T, k0, A0, rng)`
(`rbc/src/simulate.jl`) drives either through identical shock draws — pass
the same `Xoshiro(seed)` to both for a like-for-like comparison. Because
every policy returns a share in `(0, 1)`, next-period capital is positive by
construction and the simulated state is never clamped, so the out-of-bounds
diagnostics computed from the paths stay meaningful.

```julia
# from rbc/
include("src/FullRBC.jl"); using .FullRBC, Random

p = RBCParams(beta=0.98, delta=0.08, rho=0.88, gamma=3.0)
ti_policy = solve(TISolver(p))                      # Coleman time iteration
nn_solver = load_checkpoint("rbc_nn.bson")          # trained network

ti_res = simulate(ti_policy; T=200, rng=Xoshiro(42))
nn_res = simulate(nn_solver, p; T=200, rng=Xoshiro(42))
gap_metrics(nn_res, ti_res)
```

---

## Fix applied: proper Coleman time iteration

**Symptom.** The previous TI solver (a direct port of the Python
`rbc_TimeIter.py`) frequently failed to converge: for calibrations near the
edges of the training bounds (high `beta`, low `delta`, high `rho`) the
sup-norm update stalled around `1e-5`–`1e-4` forever, and for volatile draws
the policy collapsed to a spurious `c ≈ 0` fixed point (converged consumption
shares of order `1e-3` everywhere — economically nonsensical).

**Cause.** That solver used naive successive approximation on the Euler
equation: evaluate the right-hand side at the next-period capital implied by
the *previous* policy, invert marginal utility, damp with a factor 0.5. That
map is **not a contraction** — nothing guarantees the update improves the
policy, damping only slows the oscillation down, and the `c ≈ 0` corner is an
attracting spurious fixed point of the undamped map (a near-zero consumption
guess anywhere makes `mu' = c'^(-gamma)` explode, which drags the whole grid's
implied target toward zero on the next sweep).

**Fix** (`rbc/src/time_iteration.jl`). Apply the textbook Coleman operator: at
every grid node, hold the future policy fixed and **solve** the Euler equation

```text
u'(c) = beta * E[ u'(c'(k', A')) * R(k', A') ],   k' = resources - c
```

for `c` by bisection. The left side is strictly decreasing in `c` and the
right side strictly increasing (higher `c` → lower `k'` → higher return,
lower future consumption), so the gap function has a unique root and the
bracket is guaranteed: the gap `→ +∞` as `c → 0` and `→ -∞` as
`c → resources`. Under standard monotonicity/concavity conditions the Coleman
operator is a contraction, so the outer loop needs **no damping** and
converges globally.

Verified on the calibrations that previously failed (`tol = 1e-7` sup-norm on
the share, single machine, 4 threads):

| calibration                                             | before                             | after                     |
|---------------------------------------------------------|------------------------------------|---------------------------|
| `beta=0.99, delta=0.02, rho=0.99, gamma=4.0, sigma=0.05` | no convergence (diff 4e-5 @ 1000)  | converged, 279 iterations |
| random draw (`beta=0.95, rho=0.98, sigma=0.04`)          | no convergence + collapse to `c≈0` | converged, 30 iterations  |
| `alpha=0.45, rho=0.99, gamma=0.5, sigma=0.05`            | degenerate shares (1e-6 … 0.99)    | converged, interior policy |
| default / `compare.jl` calibrations                      | converged (slowly, damped)         | converged, ~2x fewer sweeps |

The per-node solves are independent, so each sweep is threaded
(`Threads.@threads`); run with `julia -t auto`. `solve` returns a `TIPolicy`
carrying `converged`, `iterations`, and the final `residual`, and the
divergence sweep records `ti_converged` per case — a benchmark that did not
converge is flagged instead of silently trusted.

## Fix applied: over-saving (transversality) penalty in the NN loss

**Symptom.** With a pure Euler-residual loss, long training runs drifted to a
severe under-consumption policy: train/val MSE kept falling to ~1e-5 while
panel rollouts showed NN consumption at ~10% of TI and capital ~10x TI, on
*every* panel case, with the drift slow and monotone (and early stopping
never triggering, because the validation loss — the same residual objective —
genuinely kept improving).

**Cause.** The Euler equation is only a first-order condition; the true
policy is pinned down by it *plus* the transversality condition. A continuum
of **over-saving** policies — consume a vanishing share, accumulate capital
without bound, consumption growth tracking `(beta*R)^(1/gamma)` — satisfies
the Euler equation pointwise and violates transversality. Because the
training residual is computed with the network supplying its own
continuation values (increasingly at `k'` far outside the training box,
where nothing constrains it), these spurious solutions score just as well as
the true one, and the loss surface is nearly flat along the "how much do we
save" direction. This is *not* multiple equilibria in the model (the RBC
model here has a unique interior equilibrium, and grid-based TI cannot fall
for the spurious solutions because its state space is compact — continuation
queries are clamped to the grid). It is an identification failure of the
pure Euler-residual objective on an unbounded state space.

**Fix** (`euler_terms` / `euler_loss` in `rbc/src/model.jl`): add an
over-saving penalty that makes the state box effectively invariant,

```julia
loss = mean(resid.^2) + k_oob_weight * mean(k_oob)
k_oob = max(k1_norm - 1, 0)^2 + max(-k1_norm, 0)^2   # k1_norm = normalized k'
```

Over-saving solutions *must* drive `k'` out of the box, so the penalty
removes them from the feasible set; inside the box it is identically zero, so
the in-box Euler dynamics are untouched. Crucially, the penalty references
only the model's own state bounds — **not** the TI benchmark — so the network
still has to learn the RBC solution on its own; the TI panel logged during
training remains diagnostic only and plays no role in stopping or model
selection. Early stopping keys on the penalized validation loss. The weight
is `--k-oob-weight` (default `1.0`); mild out-of-box excursions for extreme
calibrations (whose true ergodic set exceeds the fixed box) trade a small
boundary bias for identification.

## Fix kept: normalized Euler residual (NN training loss)

The raw Euler residual `E[beta * mu' * R'] - mu` is in marginal-utility
units, whose magnitude varies by orders of magnitude across the sampled
`(gamma, c)` range. An unnormalized MSE is dominated by whichever draws have
the most extreme curvature — starving exactly the hard high-`beta` /
low-`delta` / high-`gamma` region of gradient signal (where the Python
version's divergence sweep found its worst cases). The training loss
therefore uses the **relative** residual (`compute_residuals`):

```julia
resid = (E[beta * mu' * R'] - mu) / mu
```

so every parameter draw contributes on a comparable, dimensionless scale.

---

## Neural solver

- Inputs (8, all normalized to `[0,1]`): `k, A, alpha, beta, delta, rho,
  gamma, sigma_eps`; output: consumption share via `sigmoid`.
- Architecture: `Dense(8→64, elu) ×4 → Dense(64→1, sigmoid)`, output bias
  pre-set so the untrained policy starts at the steady-state share.
- Expectation by 7-point Gauss-Hermite quadrature (`Quadrature` in
  `rbc/src/engine/quadrature.jl`, shared with the TI solver).
- Loss: mean squared normalized Euler residual + over-saving penalty (see the
  fix sections above); early stopping on the same penalized loss evaluated on
  a fixed validation batch. A fixed NN-vs-TI panel (TI policies solved once
  up front, in parallel) is re-simulated every `eval_every` epochs — purely
  as a logged diagnostic, never as a training signal.
- Checkpointing: `Flux.state(model)` + `RBCParams` in one BSON file. The
  checkpoint format is unchanged from earlier versions, so existing
  `rbc_nn.bson` files load as-is.

---

## Workflow

All commands assume you're in `rbc/`. Instantiate once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Train

```bash
julia --project=. -t auto scripts/train.jl
```

Flags: `--batch-size` (default: 2048 on CPU, 32768 on GPU), `--epochs 50000`, `--eval-every 200`,
`--val-batch-size 8192`, `--patience 20`, `--min-rel-improve 5e-3`,
`--k-oob-weight 1.0`, `--panel-n-cases 4`, `--panel-T 120`,
`--panel-seed 321`, `--seed 42`, `--checkpoint rbc_nn.bson`,
`--device auto` (`auto`/`cpu`/`gpu`; see [GPU support](#gpu-support)).
Writes `rbc_nn.bson` and `learn_rbc_loss.png`.

### Compare one calibration

```bash
julia --project=. scripts/compare.jl
```

Writes `rbc_comparison.png`, `rbc_paths.csv`, `rbc_metrics.json`. Edit
`get_calibration_params()` in the script to change the calibration (it must
lie inside the bounds the checkpoint was trained on).

### Sweep calibrations, rank divergence

```bash
julia --project=. -t auto scripts/diagnose_divergence.jl --n-cases 40 --top-k 5 --T 200
```

Writes, under `simulation/` (or `--output-dir`), ordered by divergence rank:
`divergence_summary.csv`, `divergence_top_cases.json`, and per-case
`divergence_case_XXX_{paths.csv,plot.png}`.

### Render the notebook

```bash
quarto render main.qmd
```

Loads the trained checkpoint, simulates NN vs TI at one calibration, and
writes `main.html`.

---

## Interpreting diagnostics

- `rmse` — absolute path error; `nrmse_vs_ti_std = rmse / std(TI)` —
  scale-free error (population std, matching `numpy.std`); `level_ratio` —
  `mean(NN) / mean(TI)`.
- `mean_nrmse` / `max_nrmse` aggregate across series;
  `score_max_nrmse` ranks the sweep.
- `ti_converged` — whether the TI benchmark reached tolerance for that case;
  treat metrics with `ti_converged = false` as unreliable.
- `nn_k_oob_frac` / `ti_k_oob_frac` (and `_a_` analogues) — share of simulated
  periods outside the solver's `(k, A)` support.

Reading guide:

- `nrmse_A = 0` with overlapping TFP lines confirms both solvers saw
  identical shocks (productivity is exogenous).
- Large `nrmse_k` with a monotonically growing (not oscillating) diff is a
  persistent NN policy bias compounding over the rollout, not noise.
- High OOB fractions mean the dynamics genuinely leave the fixed
  `k_bounds = (0.5, 1.5) * k_ss(A=1)` box. For extreme calibrations (high
  persistent productivity) even the true optimal policy wants capital outside
  that box — a structural limitation of a fixed relative box, not a bug.

## Known limitations

- `sample_batch` oversamples difficult parameter corners according to
  `hard_region_prob`: high `beta`, low `delta`, and `gamma` drawn from the
  outer 20% at either edge of its normalized range.
- The NN's simulated state is intentionally never clamped between periods; a
  state outside the box the network was normalized on extrapolates without a
  restoring force (the sigmoid share keeps it bounded, but not accurate).
  The over-saving penalty discourages the *policy* from steering out of the
  box, but shock realizations can still push the state out for volatile
  calibrations.
- The training validation panel (default 4 cases) is a uniform draw and may
  miss the hard region; increase `--panel-n-cases` for model selection if you
  see divergence concentrated there.

## Lessons from experiments

- Matching training dynamics and simulation dynamics matters a lot.
  - When training residual transitions were overly clamped but simulation was not, divergence worsened.
- Aggressive targeted oversampling (high-`beta`, low-`delta`) can introduce systematic level bias.
- Wider/deeper nets alone do not guarantee better fit if objective/sampling are misaligned.
- Diagnostics that include path-level artifacts are essential to avoid being misled by scalar loss alone.

---

## GPU support

NN training runs on a GPU when one is available, automatically:

- **Detection.** `FullRBC.jl` loads the platform GPU package at module load if
  it is installed (Metal.jl on macOS/Apple GPUs, CUDA.jl elsewhere/NVIDIA);
  `select_device()` then returns the first *functional* backend via Flux's
  `gpu_device()`, falling back to the CPU. Override per run with
  `--device auto|cpu|gpu` on `scripts/train.jl` (`gpu` errors if none is
  functional, instead of silently using the CPU).
- **Precision.** The entire NN stack — network weights, training batches,
  loss, checkpoints — is uniformly **Float32 on every device** (CPU included).
  Apple GPUs have no Float64 at all, NVIDIA consumer cards run it at 1/64
  rate, and Float32 resolution (~1e-7) is far below the converged
  Euler-residual loss (~1e-4), so nothing scientific is lost and there is no
  conversion layer anywhere. The precision-sensitive numerics stay Float64:
  the **TI benchmark** (Coleman bisection converges to 1e-12 relative
  tolerance, below `eps(Float32)`) and the **simulated state paths**;
  `NNPolicy` casts the state to Float32 only at the network-input boundary.
  Batches are sampled on the CPU (reproducible per seed) and moved with
  `device(batch)`.
- **What stays on CPU.** Simulation (`NNPolicy` takes a CPU copy of the
  network — pointwise rollouts on a GPU would be slower than the copy), the
  TI benchmark, and checkpoints: `save_checkpoint` always writes CPU
  `Flux.state` (Float32); older Float64 checkpoints load transparently.
  `load_checkpoint` defaults to the CPU; pass `device=select_device()` to
  continue training on GPU.
- **Throughput / utilization.** The default network is tiny (4×64), so GPU
  training is kernel-launch-bound, not compute-bound — at small batches a
  discrete GPU sits mostly idle between launches (~20% utilization observed
  on an RTX 4090 at `--batch-size 2048` before the fixes below). Two things
  keep the GPU busy:
  1. All 7 Gauss-Hermite quadrature nodes are evaluated in **one** network
     call per step (an `8×(batch*7)` input built in `euler_terms`), instead
     of 7 separate small forward/backward passes.
  2. The default batch size is **device-dependent** (`default_batch_size`):
     2048 on CPU, 32768 on GPU. The loss is an expectation over uniform
     draws, so a larger batch only lowers gradient noise — but budget epochs
     accordingly: at 16x the batch each step sees 16x the samples, so use
     proportionally fewer epochs (e.g. `--epochs 5000` rather than 50000).
  Training is Float32 on every device, so `--device cpu` differs from GPU
  runs only in speed (and kernel-level summation order), not in precision.
- **RunPod / ephemeral containers.** Only `/workspace` survives a pod
  restart; a Julia installed via `juliaup` into `~` (and the `~/.julia`
  package depot) is wiped with the container. Install the Julia tarball
  under `/workspace` and point the depot there too:

  ```bash
  export JULIA_DEPOT_PATH=/workspace/.julia
  curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.1-linux-x86_64.tar.gz | tar xz -C /workspace
  /workspace/julia-1.12.1/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

  (re-set the `export` per session, or add it to the pod's start command).

**Troubleshooting: "No functional GPU backend found! Defaulting to CPU" on an
NVIDIA machine.** With `--device auto` training silently continues on the CPU
(run with `--device gpu` to fail loudly instead). Check, in order:

1. `nvidia-smi` inside the container/machine — if it fails, the GPU is not
   visible (e.g. the container is missing `docker run --gpus all`); nothing
   in Julia can fix that.
2. **`Pkg.add("cuDNN")` — required, not optional.** Flux's `gpu_device()`
   (MLDataDevices) only reports the CUDA backend as available when *both*
   CUDA.jl and cuDNN.jl are loaded, even though this Dense-only model never
   calls a cuDNN kernel. Without cuDNN you get the confusing state where
   `CUDA.functional()` is `true` but `gpu_device()` still returns the CPU
   (observed on a RunPod NVIDIA pod). `FullRBC.jl` picks cuDNN up
   automatically once it is installed.
3. If the precompile log showed `Failure artifact: CUDA_Runtime` and/or
   warnings about runtime libraries "loaded from a system path"
   (`LD_LIBRARY_PATH` contains `/usr/local/cuda/lib64`), point CUDA.jl at the
   system toolkit instead of its downloadable artifact:
   `julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'`
   then restart Julia.

Verify with
`julia --project=. -e 'using CUDA, cuDNN, Flux; @show CUDA.functional() gpu_device()'` —
it should print `CUDADevice(...)`, not `CPUDevice`.

---

## Dependencies

- Julia 1.10+ (developed on 1.12)
- `Flux` — policy network + training
- `FastGaussQuadrature` — Gauss-Hermite nodes/weights
- `Interpolations` — cubic B-spline TI policy
- `CairoMakie` — all figures
- `CSV`, `DataFrames`, `JSON` — tabular/metrics outputs
- `BSON` — checkpoint persistence
- `Metal` / `CUDA` — optional GPU backends, loaded only on the matching
  platform (drop the one you don't need with `Pkg.rm`)

All pinned in `rbc/Project.toml` / `rbc/Manifest.toml`.

---

## Validation performed

- Coleman TI: converges on the default and `compare.jl` calibrations, on the
  hard corner `(beta=0.99, delta=0.02, rho=0.99, gamma=4.0, sigma_eps=0.05)`,
  and on random panel draws that previously failed or collapsed (see table
  above), always to interior share policies.
- Training: gradient path through `compute_residuals` verified; short runs
  show smoothly decreasing train/val loss; early stopping, best-model restore,
  and checkpoint save/load round-trip exercised.
- The engine/model split was verified against the pre-refactor code: the
  existing `rbc_nn.bson` checkpoint loads unchanged and reproduces
  bit-identical simulated paths and loss values.
- Full 50k-epoch training was **not** re-run here; the checkpoint format and
  training objective are unchanged, so existing checkpoints remain valid.
  Retrain with `scripts/train.jl` to regenerate from scratch — panel
  diagnostics during training are now trustworthy because the TI benchmark
  they compare against actually converges.

---

## Archived Python implementation

`archive/full-rbc/` holds the original Python version (`learn_rbc.py`,
`rbc_TimeIter.py`, `compare_rbc.py`, `diagnose_divergence.py`) that the Julia
code started from; its dependencies are pinned in the top-level
`requirements.txt`. It is kept for reference only — the TI convergence and
over-saving fixes documented above were applied in the Julia version and are
**not** backported.

---

## Roadmap

- [x] Check why divergence occurs in the RBC training runs — traced to the
  over-saving identification failure and a non-contracting TI benchmark; see
  the "Fix applied" sections.
- [x] Isolate the model-agnostic training engine (`rbc/src/engine/`) from the
  model-specific code (`rbc/src/model.jl`) so the loss/model can be swapped
  without touching the engine.
- [ ] Add a second model on the engine (RBC with labor/leisure choice:
  two-output policy, intratemporal FOC in the loss).
- [ ] Extend to richer DSGE settings (NK with Rotemberg pricing and a ZLB;
  heterogeneous agents will need a separate engine variant).
- [ ] Introduce rollout-aware training terms for long-horizon dynamic consistency.
- [ ] Improve ranking diagnostics with blended scores (normalized + level + correlation).
- [ ] Standardize experiment tracking across model/seed/config runs.

---

## License

MIT
