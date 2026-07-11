"""
    FullRBC

Stochastic RBC model solved two ways:

- [`NNSolver`](@ref): a neural-network consumption-share policy trained on the
  normalized Euler-equation residual over a whole box of structural
  parameters, and
- [`TISolver`](@ref)/[`solve`](@ref): a Coleman time-iteration benchmark
  (exact per-node Euler solve + cubic-spline policy) at a single calibration.

Both produce policies implementing `consumption_share(policy, k, A)`, so one
shared [`simulate`](@ref) drives either through identical shock draws for
like-for-like comparison ([`gap_metrics`](@ref)).

Started as a port of the Python `full-rbc` package; see `README.md` for the
design and for the fixes applied relative to that version.
"""
module FullRBC

using Random
using Statistics
using Flux
using FastGaussQuadrature
using Interpolations
using BSON

export RBCParams, steady_state, steady_state_batch, steady_state_share,
       a_support_from_shock_params, k_support, Quadrature,
       production, resources, gross_return, marginal_utility, next_productivity,
       sample_params_uniform, with_calibration, params_to_dict, params_from_dict,
       TISolver, TIPolicy, solve,
       NNSolver, NNPolicy, sample_batch, compute_residuals, euler_terms, euler_loss, train!,
       save_checkpoint, load_checkpoint,
       consumption_share, simulate,
       gap_metrics, build_validation_panel, evaluate_validation_panel

include("model.jl")
include("time_iteration.jl")
include("neural_solver.jl")
include("simulate.jl")
include("diagnostics.jl")

end # module
