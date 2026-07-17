"""
    FullRBC

Stochastic RBC models solved by neural networks against Coleman time-iteration
benchmarks. Two models share one training engine:

- [`RBCParams`](@ref) — the baseline model: consumption-share policy trained
  on the normalized Euler residual ([`NNSolver`](@ref)) vs [`TISolver`](@ref).
- [`RBCLaborParams`](@ref) — RBC with endogenous labor (CRRA + separable
  isoelastic disutility): an hours policy trained on the substituted Euler
  residual (the intratemporal condition is imposed exactly, pinning
  consumption in closed form), vs [`LaborTISolver`](@ref).

Within a model, NN and TI policies implement one query interface
(`consumption_share` / `controls`), so a shared [`simulate`](@ref) drives
either through identical shock draws for like-for-like comparison
([`gap_metrics`](@ref)).

The code is split into a model-agnostic engine (`engine/`: device selection,
network builder, training loop, checkpointing, metrics) and the RBC model
(`model.jl`, plus its TI benchmark and simulate loop). The engine reaches the
economics only through the interface functions in `engine/interface.jl`,
dispatched on the parameter-struct type — adding a model, or swapping the
training loss, touches only model files.

Started as a port of the Python `full-rbc` package (now `archive/full-rbc/`);
see the repository-root `README.md` for the design and for the fixes applied
relative to that version.
"""
module FullRBC

using Random
using Statistics
using Flux
using FastGaussQuadrature
using Interpolations
using BSON

# Optional GPU backends. Flux's `gpu_device()` only sees a backend whose
# trigger package is loaded, so load the platform-appropriate one if it is
# installed: Metal on macOS (Apple GPUs), CUDA elsewhere (NVIDIA). A failure
# to load is downgraded to a warning — everything then runs on the CPU.
if Sys.isapple()
    if Base.find_package("Metal") !== nothing
        try
            @eval using Metal
        catch err
            @warn "Metal.jl is installed but failed to load; using CPU" err
        end
    end
elseif Base.find_package("CUDA") !== nothing
    try
        @eval using CUDA
        # cuDNN is only needed for conv/batchnorm kernels (not this model),
        # but loading it when installed silences Flux's FluxCUDAExt warning.
        Base.find_package("cuDNN") === nothing || @eval using cuDNN
    catch err
        @warn "CUDA.jl is installed but failed to load; using CPU" err
    end
end

export RBCParams, steady_state, steady_state_batch, steady_state_share,
       a_support_from_shock_params, k_support, Quadrature,
       production, resources, gross_return, marginal_utility, next_productivity,
       sample_params_uniform, with_calibration, params_to_dict, params_from_dict,
       TISolver, TIPolicy, solve,
       NNSolver, NNPolicy, sample_batch, compute_residuals, euler_terms, euler_loss, train!,
       policy_spec, training_loss, validation_report,
       save_checkpoint, load_checkpoint, select_device, is_cpu_device, default_batch_size,
       consumption_share, simulate,
       gap_metrics, build_validation_panel, evaluate_validation_panel,
       RBCLaborParams, LaborTISolver, LaborTIPolicy, NNLaborPolicy, controls,
       labor_terms, labor_loss

# Model-agnostic engine: nothing in engine/ references the RBC model.
include("engine/device.jl")
include("engine/quadrature.jl")
include("engine/interface.jl")
include("engine/network.jl")
include("engine/training.jl")
include("engine/diagnostics.jl")

# The RBC model: economics + engine-interface implementation + benchmark.
include("model.jl")
include("time_iteration.jl")
include("simulate.jl")

# Second model on the same engine: RBC with endogenous labor (CRRA + separable
# isoelastic labor disutility), self-contained in one file.
include("model_alter.jl")

end # module
