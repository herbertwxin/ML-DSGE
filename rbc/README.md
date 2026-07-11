# full-rbc-julia

A stochastic RBC model solved two ways and compared head-to-head:

- **Neural solver** (`src/neural_solver.jl`) — a policy network trained on the normalized Euler-equation residual over a whole box of structural
  parameters.
- **Time Iteration benchmark** (`src/time_iteration.jl`) — Coleman time
  iteration (exact per-node Euler solve + cubic-spline policy) at a single calibration. Solved using bisection. 

---

## Repository layout

```text
full-rbc-julia/
├── Project.toml / Manifest.toml   # pinned dependencies
├── src/
│   ├── FullRBC.jl                 # module entry point
│   ├── model.jl                   # RBCParams, technology/preferences, steady state, quadrature
│   ├── time_iteration.jl          # TISolver/TIPolicy: Coleman time iteration
│   ├── neural_solver.jl           # NNSolver/NNPolicy: policy net + Euler-residual training
│   ├── simulate.jl                # one simulate() shared by every policy type
│   └── diagnostics.jl             # NN-vs-TI gap metrics
├── scripts/
│   ├── common.jl                  # shared CLI/IO/Makie-figure helpers
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

## Model

- State: capital `k`, productivity `A`; control: consumption `c`.
- Preferences: CRRA, `u'(c) = c^(-gamma)`.
- Technology: `y = A k^alpha`, depreciation `delta`.
- Productivity: `log A' = rho log A + sigma_eps * eps`, `eps ~ N(0,1)`.
- Structural parameters `(alpha, beta, delta, gamma, rho, sigma_eps)` live in
  `RBCParams` (`src/model.jl`) together with the training bounds and
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

A single `simulate(policy, p::RBCParams; T, k0, A0, rng)` (`src/simulate.jl`)
drives either through identical shock draws — pass the same `Xoshiro(seed)`
to both for a like-for-like comparison. Because every policy returns a share
in `(0, 1)`, next-period capital is positive by construction and the
simulated state is never clamped, so the out-of-bounds diagnostics computed
from the paths stay meaningful.

```julia
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

**Symptom.** The previous TI solver (a direct port of `rbc_TimeIter.py`)
frequently failed to converge: for calibrations near the edges of the
training bounds (high `beta`, low `delta`, high `rho`) the sup-norm update
stalled around `1e-5`–`1e-4` forever, and for volatile draws the policy
collapsed to a spurious `c ≈ 0` fixed point (converged consumption shares of
order `1e-3` everywhere — economically nonsensical).

**Cause.** That solver used naive successive approximation on the Euler
equation: evaluate the right-hand side at the next-period capital implied by
the *previous* policy, invert marginal utility, damp with a factor 0.5. That
map is **not a contraction** — nothing guarantees the update improves the
policy, damping only slows the oscillation down, and the `c ≈ 0` corner is an
attracting spurious fixed point of the undamped map (a near-zero consumption
guess anywhere makes `mu' = c'^(-gamma)` explode, which drags the whole grid's
implied target toward zero on the next sweep).

**Fix** (`src/time_iteration.jl`). Apply the textbook Coleman operator: at
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

**Fix** (`euler_terms` / `euler_loss` in `src/neural_solver.jl`): add an
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
  `src/model.jl`, shared with the TI solver).
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

Flags: `--batch-size 2048`, `--epochs 50000`, `--eval-every 200`,
`--val-batch-size 8192`, `--patience 20`, `--min-rel-improve 5e-3`,
`--k-oob-weight 1.0`, `--panel-n-cases 4`, `--panel-T 120`,
`--panel-seed 321`, `--seed 42`, `--checkpoint rbc_nn.bson`.
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

- `hard_region_prob = 0.0` by default: the oversampling mechanism for the
  hard (high `beta`, low `delta`) region exists in `sample_batch` but is
  inert unless enabled via `RBCParams(hard_region_prob=0.2, ...)`.
- The NN's simulated state is intentionally never clamped between periods; a
  state outside the box the network was normalized on extrapolates without a
  restoring force (the sigmoid share keeps it bounded, but not accurate).
  The over-saving penalty discourages the *policy* from steering out of the
  box, but shock realizations can still push the state out for volatile
  calibrations.
- The training validation panel (default 4 cases) is a uniform draw and may
  miss the hard region; increase `--panel-n-cases` for model selection if you
  see divergence concentrated there.

---

## Dependencies

- Julia 1.10+ (developed on 1.12)
- `Flux` — policy network + training
- `FastGaussQuadrature` — Gauss-Hermite nodes/weights
- `Interpolations` — cubic B-spline TI policy
- `CairoMakie` — all figures
- `CSV`, `DataFrames`, `JSON` — tabular/metrics outputs
- `BSON` — checkpoint persistence

All pinned in `Project.toml` / `Manifest.toml`.

---

## Validation performed

- Coleman TI: converges on the default and `compare.jl` calibrations, on the
  hard corner `(beta=0.99, delta=0.02, rho=0.99, gamma=4.0, sigma_eps=0.05)`,
  and on random panel draws that previously failed or collapsed (see table
  above), always to interior share policies.
- Training: gradient path through `compute_residuals` verified; short runs
  show smoothly decreasing train/val loss; early stopping, best-model restore,
  and checkpoint save/load round-trip exercised.
- The pre-existing `rbc_nn.bson` checkpoint loads unchanged and
  `scripts/compare.jl` / `scripts/diagnose_divergence.jl` run end-to-end on
  it, producing the Makie figures and CSV/JSON outputs.
- Full 50k-epoch training was **not** re-run here; the checkpoint format and
  training objective are unchanged, so existing checkpoints remain valid.
  Retrain with `scripts/train.jl` to regenerate from scratch — panel
  diagnostics during training are now trustworthy because the TI benchmark
  they compare against actually converges.

## License

MIT (same as the parent repository).
