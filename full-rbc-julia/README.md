# full-rbc-julia

Julia port of [`full-rbc`](../full-rbc) — a stochastic RBC model solved by:
- a neural-network policy trained on the Euler-equation residual (`src/neural_solver.jl`), and
- a Time Iteration (cubic-spline) benchmark (`src/time_iteration.jl`).

This port applies one substantive fix on top of the direct translation: the
training loss now uses a **normalized Euler residual** instead of the raw,
marginal-utility-scale residual used in the Python version. See
[Fix applied](#fix-applied-normalized-euler-residual) below for the motivation
and diagnosis that led to this change.

---

## Why a Julia port

Same model, same algorithm, same diagnostics — just written for someone more
comfortable in Julia than in Python/PyTorch. Where a straight 1:1 translation
would have been awkward, idiomatic Julia was used instead (see
[Differences from the Python version](#differences-from-the-python-version)),
but the economics, the numerical methods, and the file/output layout are kept
as close as possible to `full-rbc` so results are directly comparable.

---

## Repository layout

```text
full-rbc-julia/
├── Project.toml / Manifest.toml   # pinned dependencies
├── src/
│   ├── FullRBC.jl                 # module entry point
│   ├── params.jl                  # RBCParams, steady state, A-support helper
│   ├── time_iteration.jl          # TISolver: cubic-spline Time Iteration
│   ├── neural_solver.jl           # NNSolver: policy net + Euler-residual training
│   └── diagnostics.jl             # shared NN-vs-TI gap metrics
├── scripts/
│   ├── train.jl                   # train the NN checkpoint
│   ├── compare.jl                 # single-calibration NN vs TI comparison
│   └── diagnose_divergence.jl     # parameter-sweep divergence diagnostics
├── simulation/                    # generated divergence outputs (csv/json/png)
├── rbc_nn.bson                    # trained checkpoint (generated)
├── rbc_comparison.png             # comparison figure (generated)
├── rbc_paths.csv                  # time paths (generated)
└── rbc_metrics.json               # metrics summary (generated)
```

---

## RBC model (unchanged from the Python version)

### States / control
- State: capital `k`, productivity `A`
- Control: consumption `c`

### Structural parameters
- `alpha`: capital share
- `beta`: discount factor
- `delta`: depreciation rate
- `gamma`: CRRA risk aversion
- `rho`: productivity persistence
- `sigma_eps`: shock innovation std

Productivity law of motion: `log A' = rho * log A + sigma_eps * eps`, `eps ~ N(0,1)`.

---

## Solver design

### 1) Neural solver (`src/neural_solver.jl`)

Policy network predicts consumption share from normalized inputs:
- Inputs (8D): `(k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm)`
- Output: `frac ∈ (0,1)` via `sigmoid`
- Consumption: `c = frac * resources`, `resources = A*k^alpha + (1-delta)*k`

Architecture: `Dense(8,64,elu) → Dense(64,64,elu) → Dense(64,64,elu) → Dense(64,64,elu) → Dense(64,1,sigmoid)`,
with the output bias pre-initialized so the untrained policy starts near the
steady-state consumption share (`build_policy_net`).

Training objective: normalized Euler-equation residual MSE, expectation taken
by 7-point Gauss-Hermite quadrature (`compute_residuals`).

Important implementation details (same as Python):
- Dynamic productivity normalization uses the stationary log-TFP scale:
  `sigma_stat = sigma_eps / sqrt(1 - rho^2)`, `A` support `≈ exp(± A_sigma_mult * sigma_stat)`.
- Training uses early stopping on a fixed validation residual batch.
- A fixed TI validation panel (`build_validation_panel`) is solved once and
  reused for cheap NN-vs-TI diagnostics during training.
- No clamping of `(k, A)` between periods in `simulate`, matching the training
  dynamics (this is unchanged from Python; see [Known limitations](#known-limitations-carried-over-from-python)).

### 2) Time Iteration benchmark (`src/time_iteration.jl`)

- Cubic B-spline policy over a `(k, A)` grid (Interpolations.jl), with linear
  extrapolation outside the grid — the idiomatic-Julia analogue of SciPy's
  unclamped `RectBivariateSpline` extrapolation used in the Python version.
- Same `RBCParams` object as the NN solver, so the two are directly comparable
  at any calibration.
- Same shock seed (`Xoshiro(seed)`, passed explicitly) is used for NN vs TI
  comparisons, replacing Python's pattern of resetting `np.random.seed()`
  before each `.simulate()` call.

---

## Fix applied: normalized Euler residual

Diagnosing the Python version's `divergence_top_cases.json` showed:
- Divergence concentrated in **capital**, growing as a **persistent, monotonic
  bias** (not chaotic noise) over the simulation horizon.
- The consumption policy was already slightly off even at `t=0` (same state
  for NN and TI) — i.e. a genuine, if small, policy-function approximation
  error, not just accumulated simulation drift.
- The worst cases clustered in a specific corner of parameter space: **high
  beta, low delta, high rho** (near-unit-root capital dynamics) and
  **moderate-to-high gamma**. In that corner, small per-period policy errors
  don't mean-revert — they integrate into large capital gaps.

One concrete, fixable contributor: the Python training loss used the **raw**
Euler residual,

```text
resid_raw = beta * E[mu' * R'] - mu,      mu = c^(-gamma)
loss      = mean(resid_raw^2)
```

Because training draws `(alpha, beta, delta, rho, gamma, sigma_eps)` uniformly
from very wide bounds, `mu`'s magnitude varies by **orders of magnitude**
across sampled draws (higher `gamma` and/or lower `c` blow it up). An
unnormalized MSE loss is dominated by whichever sampled draw happens to have
the largest `|mu|`, which has nothing to do with which region of parameter
space is economically "hard" to fit — likely under-training exactly the
high-beta/low-delta/high-gamma corner where the divergence sweep found its
worst cases.

**Fix** (`src/neural_solver.jl`, `compute_residuals`):

```julia
resid_raw = expected_rhs .- mu
resid     = resid_raw ./ (mu .+ 1e-8)   # relative Euler residual
```

Dividing by the target marginal utility turns the residual into a
dimensionless, *relative* Euler-equation error (economically: the fractional
deviation of today's marginal utility from its consumption-smoothed optimum).
Every sampled parameter draw now contributes to the loss on a comparable
scale, regardless of `gamma`/`c` magnitude, so gradient steps are no longer
dominated by whichever draw happens to have extreme curvature.

This is a training-time change only; the TI benchmark, the model equations,
and the simulation/comparison logic are otherwise numerically identical in
spirit to the Python version.

---

## Fix applied: TI benchmark extrapolation could explode

While validating this port, the training-time NN-vs-TI validation panel
(`panel_n_cases=4`, `panel_T=120`) occasionally showed wildly pathological
metrics, e.g. `level_ratio_c` in the tens of thousands or negative — clearly
not a sane "NN is a bit off" signal. Isolating it (see git history / issue
discussion) showed the problem was in the **TI benchmark's own rollout**, not
in the NN or the training loss (`train_mse`/`val_mse` were small and
decreasing smoothly throughout):

- `TISolver`'s cubic-spline policy interpolates a raw, **unconstrained
  consumption level** (unlike the NN, whose sigmoid output always bounds the
  consumption *share* to `(0,1)`).
- For calibrations near the edges of the training bounds (high `sigma_eps`,
  high `rho`, and/or high `gamma`), a 120-period rollout can legitimately push
  `(k, A)` outside the TI grid `(k_min,k_max) × (A_min,A_max)`.
- Once outside, unclamped `Interpolations.jl` linear extrapolation of the
  spline has no guarantee of respecting the budget constraint. Even just
  *clamping the query point* (evaluate `policy(clamp(k,...), clamp(A,...))`)
  turned out to be insufficient: reusing the boundary's absolute consumption
  **level** becomes a vanishing share of the true (much larger) resources as
  `k` drifts further out, which *guarantees* unbounded one-directional capital
  growth every subsequent period — a self-reinforcing numerical explosion
  entirely separate from anything the NN does.

**Fix** (`src/time_iteration.jl`, `simulate(::TISolver, policy; ...)`): outside
the fitted grid, convert the boundary lookup into an **implied consumption
share** and apply that share to the true resources at the actual `(k, A)`:

```julia
kt_q, At_q = clamp(kt, ti.k_min, ti.k_max), clamp(At, ti.A_min, ti.A_max)
c_boundary = policy(kt_q, At_q)
resources_boundary = At_q * kt_q^p.alpha + (1 - p.delta) * kt_q
frac = clamp(c_boundary / resources_boundary, 1e-6, 1.0 - 1e-6)
ct = frac * resources   # resources computed at the TRUE (kt, At)
```

Inside the grid this is mathematically identical to `ct = policy(kt, At)` (no
behavior change there). Outside the grid it gives the TI benchmark the same
kind of self-correcting, share-based extrapolation the NN gets for free from
its sigmoid output, instead of an explosive absolute-level continuation. The
*physical* state series returned by `simulate` is still never clamped, so
`ti_k_oob_frac`/`ti_a_oob_frac` remain meaningful diagnostics of how often a
given calibration's dynamics genuinely leave the training box.

**What this does *not* fix**: for calibrations at the extreme edges of the
training bounds, the fixed relative box `k_bounds = (0.5, 1.5) * k_ss(A=1)`
can simply be too narrow for the model's own ergodic dynamics at high
persistent productivity — even the *true* optimal policy would want capital
outside that box there. High `ti_k_oob_frac`/`nn_k_oob_frac` for such draws is
an expected structural limitation of a fixed-relative-to-unconditional-SS box
(shared with the Python version), not a numerical bug; see
[Troubleshooting](#troubleshooting) below for how this shows up during
training.

---

## Differences from the Python version

Deliberate departures from a literal 1:1 translation, made because they're
more idiomatic in Julia (behavior should be equivalent unless noted):

- **No global RNG mutation.** Python resets `np.random.seed(seed)` before each
  `.simulate()` call so NN and TI draw the same shocks. Julia instead passes
  an explicit `rng::AbstractRNG` (typically `Xoshiro(seed)`) into `simulate`,
  which is safer under threading and avoids hidden global state.
- **Threads instead of processes.** `diagnose_divergence.py` uses
  `ProcessPoolExecutor`/`--n-workers`; the Julia version uses
  `Threads.@threads` over independent cases. Control parallelism with
  `julia -t auto` (or `-t N`) instead of a `--n-workers` flag.
- **`gap_metrics` is a single shared function** (`src/diagnostics.jl`) used by
  training, `compare.jl`, and `diagnose_divergence.jl`, instead of two
  near-duplicate Python implementations (`RBCSolver._gap_metrics` and
  `diagnose_divergence.compute_gap_metrics`).
- **`steady_state_batch`** works on scalars *or* arrays via broadcasting,
  replacing the separate scalar (`RBCTISolver.calculate_steady_state`) and
  tensor (`RBCSolver._steady_state_batch`) implementations in Python.
- **Cubic spline library**: Interpolations.jl (native Julia) instead of
  SciPy's FITPACK-based `RectBivariateSpline`. Both produce smooth cubic
  interpolants with linear-ish extrapolation outside the grid, but they are
  not bit-for-bit identical numerically.
- **Model persistence**: a single BSON file (`rbc_nn.bson`) holding
  `Flux.state(model)` plus the `RBCParams` struct directly (no need for the
  dict round-trip Python uses for `torch.save`); `params_to_dict`/`params_from_dict`
  are kept only for the JSON outputs (`rbc_metrics.json`, `divergence_top_cases.json`).
- **CPU only.** The Python version picks CUDA/MPS/CPU via `get_device()`. This
  port targets CPU (via Flux + OpenBLAS); given the small model (4×64 hidden
  units) and modest batch sizes, CPU is fast enough for this problem. Adding
  GPU support (CUDA.jl/Metal.jl) later is straightforward if needed — swap the
  model/data to a `CuArray`/`MtlArray` device and the rest of the code (which
  is written with broadcasting, not manual loops) should carry over largely
  unchanged.
- **No explicit `--train-if-missing` / `--n-workers` CLI flags.** `scripts/diagnose_divergence.jl`
  and `scripts/compare.jl` simply error out with an instructive message if
  `rbc_nn.bson` is missing; run `scripts/train.jl` first.

## Known limitations carried over from Python

These were flagged as likely secondary contributors to divergence but are
**not** changed in this port, to keep the fix isolated and easy to evaluate on
its own. They're good candidates for follow-up experiments:

- `hard_region_prob = 0.0` by default: the oversampling mechanism for the
  "high beta / low delta" hard region (`sample_batch`) exists but is inert
  unless you explicitly set `RBCParams(hard_region_prob=0.2, ...)`.
- No clamping of `(k, A)` between periods during `simulate` or during the
  recursive next-state construction in `compute_residuals` — once a simulated
  state drifts outside the box the network was normalized on, it can
  extrapolate without a restoring force.
- The training validation panel (`panel_n_cases`, default 4) is drawn
  uniformly at random and may not include the hard region at all; consider
  passing a larger panel or a stratified sample for model selection if you
  see the same divergence pattern here.

---

## End-to-end workflow

All commands assume you're in `full-rbc-julia/`. Instantiate the environment once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Train the NN checkpoint

```bash
julia --project=. -t auto scripts/train.jl
```

Optional flags (all have the same defaults as the Python version):
`--batch-size`, `--epochs`, `--eval-every`, `--val-batch-size`, `--patience`,
`--min-rel-improve`, `--panel-n-cases`, `--panel-T`, `--panel-seed`, `--seed`.

Writes:
- `rbc_nn.bson`
- `learn_rbc_loss.png`

### Compare one calibration (NN vs TI)

```bash
julia --project=. scripts/compare.jl
```

Writes:
- `rbc_comparison.png`
- `rbc_paths.csv`
- `rbc_metrics.json`

Edit `get_calibration_params()` in `scripts/compare.jl` to compare a different
calibration (must lie within the bounds used to train the checkpoint).

### Sweep many calibrations and diagnose divergence

```bash
julia --project=. -t auto scripts/diagnose_divergence.jl --n-cases 40 --top-k 5 --T 200
```

Outputs under `simulation/`:
- `divergence_summary.csv`
- `divergence_top_cases.json`
- `divergence_case_XXX_paths.csv`
- `divergence_case_XXX_plot.png`

---

## Interpreting diagnostics

Same metrics and reading guide as the Python version:

- `rmse`: absolute path error.
- `nrmse_vs_ti_std = rmse / std(TI series)`: scale-free error (uses the
  population std, i.e. `corrected=false`, to match `numpy.std`'s default).
- `mean_nrmse`, `max_nrmse`: aggregate diagnostics across series.
- `nn_k_oob_frac`, `ti_k_oob_frac` (and the `_a_` productivity analogues): share
  of simulated periods outside the solver's `(k, A)` support.

Practical reading guide:
- `nrmse_A = 0` and overlapping TFP lines mean the shock process alignment is
  correct (productivity is exogenous and identical given the same seed).
- Large `nrmse_k` with a **monotonically growing, not oscillating** diff
  (check the `_paths.csv`) usually indicates a persistent policy-level bias
  rather than a variance/noise mismatch — see [Fix applied](#fix-applied-normalized-euler-residual).
- High OOB fractions signal the simulated state left the box the network (or
  TI spline) was fit on, which compounds any existing bias via extrapolation.

---

## Configuration notes

Main parameter ranges live in `RBCParams` (`src/params.jl`):
- structural bounds (`alpha_bounds`, `beta_bounds`, `delta_bounds`, `rho_bounds`, `gamma_bounds`, `sigma_eps_bounds`)
- state scaling controls (`k_bounds`, `A_sigma_mult`)
- training sampling controls (`hard_region_prob`, `hard_beta_low_norm`, `hard_delta_high_norm`)

For reproducible experiments, change one mechanism at a time and compare
`simulation/divergence_summary.csv` across runs.

---

## Dependencies

- Julia 1.10+ (developed/tested on 1.12)
- `Flux` — neural network + autodiff-based training
- `FastGaussQuadrature` — Gauss-Hermite quadrature nodes/weights
- `Interpolations` — cubic B-spline policy interpolation (TI benchmark)
- `CSV`, `DataFrames` — tabular path/summary outputs
- `JSON` — metrics/diagnostics output
- `Plots` — comparison figures
- `BSON` — model checkpoint persistence

All pinned in `Project.toml`/`Manifest.toml`. Instantiate with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Troubleshooting

**"The panel `mean_nrmse`/`max_nrmse`/`level_ratio_*` numbers look insane
(hundreds or thousands) early in training — is training broken?"**

Probably not. Check `train_mse`/`val_mse` in the same log line first: if those
are small and decreasing smoothly (no `NaN`/`Inf`, no sudden jumps), the
optimizer itself is fine. The panel metrics are computed from a **120-period
rollout** (`panel_T`) at only `panel_n_cases` (default 4) randomly-drawn
calibrations, which:

1. compounds even a small per-period consumption-share error over many
   periods (a policy that's off by 1% per period can look very off after 120
   periods of compounding), and
2. can easily include one calibration near the edge of the training bounds
   (high `sigma_eps`/`rho`/`gamma`) where the `(k, A)` box is genuinely too
   narrow for the dynamics (see previous section) — a single such case can
   dominate a 4-case average.

Early in training (e.g. epoch 1,600 out of a default 50,000), the network
simply hasn't yet learned an accurate consumption share, especially in
under-sampled regions (`hard_region_prob=0.0` by default; see
[Known limitations](#known-limitations-carried-over-from-python)), so a
long rollout can show `level_ratio_k` of several times TI's mean capital even
though the Euler residual loss is healthy. Watch the trend over successive
`eval_every` checkpoints — it should be monotonically improving (it was, in
testing: `mean_nrmse` fell from ~680 to ~250 and `level_ratio_c` from ~3,260 to
~950 over the first 1,600 epochs of one run). If it plateaus at a large value
for thousands of epochs with `train_mse` still visibly high, that's a real
signal to investigate (e.g. try `--panel-n-cases` larger, or enable
`hard_region_prob`).

If you instead see `NaN`, `Inf`, or clearly nonsensical **negative**
consumption/output in `train_mse`/`val_mse` or in `rbc_paths.csv`, that *is* a
bug — please check `nn_k_oob_frac`/`ti_k_oob_frac` in the relevant summary
first (a state that has drifted far outside its solver's box is the most
likely proximate cause) and open an issue with the offending `RBCParams`.

---

## Validation performed

The full pipeline (module load → `TISolver`/`solve!`/`simulate` →
`NNSolver`/`sample_batch`/`compute_residuals` → one `Flux.withgradient` +
`Flux.update!` step → `train!` with a threaded validation panel → BSON
save/load round-trip → `scripts/compare.jl` → `scripts/diagnose_divergence.jl`
with multi-threaded case evaluation) was smoke-tested end-to-end on this
machine with short-epoch, small-panel settings to confirm every stage runs and
produces correctly-shaped outputs (CSV/JSON/PNG). It was **not** trained to
full convergence here (50,000 epochs is a ~45–60 minute CPU run); run
`scripts/train.jl` with its defaults for a real checkpoint before trusting the
comparison/diagnostic outputs.

---

## License

MIT (same as the parent repository).
