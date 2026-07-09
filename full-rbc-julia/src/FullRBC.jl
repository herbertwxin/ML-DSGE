"""
    FullRBC

Julia port of `full-rbc` (Python/PyTorch) — a stochastic RBC model solved by:
- a neural policy approximation trained on the Euler-equation residual, and
- a Time Iteration (cubic-spline) benchmark.

See `README.md` in this folder for the full write-up, including the fix applied
relative to the Python version: the Euler residual used in the training loss is
now *normalized* by the target marginal utility (see [`compute_residuals`](@ref)),
instead of being used in raw marginal-utility units.
"""
module FullRBC

using Random
using Statistics
using LinearAlgebra
using Flux
using FastGaussQuadrature
using Interpolations
using BSON

export RBCParams, a_support_from_shock_params, steady_state, steady_state_batch,
       params_to_dict, params_from_dict, sample_params_uniform,
       TISolver, solve!, simulate,
       NNSolver, sample_batch, compute_residuals, train!, save_checkpoint, load_checkpoint,
       gap_metrics, build_validation_panel, evaluate_validation_panel

include("params.jl")
include("time_iteration.jl")
include("neural_solver.jl")
include("diagnostics.jl")

end # module
