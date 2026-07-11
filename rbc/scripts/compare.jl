#=
Single-calibration NN vs TI comparison.

Usage:
    julia --project=. scripts/compare.jl

Requires a trained checkpoint at `rbc_nn.bson` (run `scripts/train.jl` first).

Writes, under the project root:
    rbc_comparison.png
    rbc_paths.csv
    rbc_metrics.json
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
include(joinpath(@__DIR__, "common.jl"))
using .FullRBC
using Random
using JSON

const T_SIM = 200
const SIM_SEED = 42

"""
Calibration used for the comparison (TI solve + both simulations). Edit to
compare a different one; it must lie inside the bounds the checkpoint was
trained on (see `RBCParams` `*_bounds` fields).
"""
get_calibration_params() = RBCParams(alpha=0.30, beta=0.98, delta=0.08, rho=0.88, gamma=3.0, sigma_eps=0.02)

function run_comparison(;
    params::RBCParams=get_calibration_params(),
    plot_file::String="rbc_comparison.png",
    paths_file::String="rbc_paths.csv",
    metrics_file::String="rbc_metrics.json",
)
    nn_solver = load_nn_solver()
    ti_policy = solve(TISolver(params); verbose=true)
    ti_policy.converged || @warn "TI benchmark did not converge; metrics may be unreliable"

    nn_res = simulate(nn_solver, params; T=T_SIM, rng=Xoshiro(SIM_SEED))
    ti_res = simulate(ti_policy; T=T_SIM, rng=Xoshiro(SIM_SEED))

    plot_path = joinpath(ROOT, plot_file)
    save(plot_path, comparison_figure(nn_res, ti_res; params))
    @info "Saved comparison figure" plot_path

    paths_path = save_path_table(nn_res, ti_res, joinpath(ROOT, paths_file))
    @info "Saved NN/TI paths" paths_path

    metrics = gap_metrics(nn_res, ti_res; series=SERIES)
    metrics_path = joinpath(ROOT, metrics_file)
    open(metrics_path, "w") do io
        JSON.print(io, Dict("params" => params_to_dict(params), "metrics" => metrics), 2)
    end
    @info "Saved NN/TI gap metrics" metrics_path

    return nn_res, ti_res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
