#=
Single-calibration NN vs TI comparison (Julia port of `compare_rbc.py`).

Usage:
    julia --project=. scripts/compare.jl

Requires a trained checkpoint at `rbc_nn.bson` (run `scripts/train.jl` first).

Writes, under the project root:
    rbc_comparison.png
    rbc_paths.csv
    rbc_metrics.json
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
using .FullRBC
using Random
using CSV
using DataFrames
using JSON
using Plots

const ROOT = dirname(@__DIR__)
const CHECKPOINT_PATH = joinpath(ROOT, "rbc_nn.bson")
const T_SIM = 200
const SIM_SEED = 42
const SERIES = (:consumption, :capital, :output, :investment, :productivity)

"""
Parameter set used for comparison (TI solve + both simulations). Edit this to
use a different calibration; it must lie within the bounds used to train the
NN checkpoint (see `RBCParams` field defaults / `*_bounds` fields).
"""
get_calibration_params() = RBCParams(alpha=0.30, beta=0.98, delta=0.08, rho=0.88, gamma=3.0, sigma_eps=0.02)

function get_nn_solver()
    isfile(CHECKPOINT_PATH) ||
        error("No checkpoint at $CHECKPOINT_PATH. Run `julia --project=. scripts/train.jl` first.")
    return load_checkpoint(CHECKPOINT_PATH)
end

function save_path_table(nn_res, ti_res, path)
    df = DataFrame(t=0:(length(nn_res.consumption)-1))
    for s in SERIES
        df[!, Symbol(s, :_nn)] = getfield(nn_res, s)
        df[!, Symbol(s, :_ti)] = getfield(ti_res, s)
        df[!, Symbol(s, :_diff)] = getfield(nn_res, s) .- getfield(ti_res, s)
    end
    CSV.write(path, df)
    @info "Saved NN/TI paths" path
end

function save_comparison_plot(nn_res, ti_res, path)
    t = 0:(T_SIM-1)
    titles = Dict(:consumption => "Consumption", :capital => "Capital", :output => "Output",
                  :investment => "Investment", :productivity => "TFP (productivity)")
    subplots = [
        plot(t, [getfield(nn_res, s) getfield(ti_res, s)], label=["NN" "TI"],
             linestyle=[:solid :dash], title=titles[s], xlabel="t")
        for s in SERIES
    ]
    fig = plot(subplots..., layout=(3, 2), size=(1000, 1000),
               plot_title="RBC: Neural network vs Time Iteration (same calibration, same seed)")
    savefig(fig, path)
    @info "Saved comparison plot" path
end

function run_comparison(;
    params::RBCParams=get_calibration_params(),
    save_plot::String="rbc_comparison.png",
    save_paths_file::String="rbc_paths.csv",
    save_metrics_file::String="rbc_metrics.json",
)
    nn_solver = get_nn_solver()
    ti = TISolver(params)
    policy = solve!(ti)

    nn_res = simulate(nn_solver; T=T_SIM, alpha=params.alpha, beta=params.beta, delta=params.delta,
                       rho=params.rho, gamma=params.gamma, sigma_eps=params.sigma_eps, rng=Xoshiro(SIM_SEED))
    ti_res = simulate(ti, policy; T=T_SIM, rng=Xoshiro(SIM_SEED))

    save_comparison_plot(nn_res, ti_res, joinpath(ROOT, save_plot))
    save_path_table(nn_res, ti_res, joinpath(ROOT, save_paths_file))

    metrics = gap_metrics(nn_res, ti_res; series=SERIES)
    out = Dict("params" => params_to_dict(params), "metrics" => metrics)
    metrics_path = joinpath(ROOT, save_metrics_file)
    open(metrics_path, "w") do io
        JSON.print(io, out, 2)
    end
    @info "Saved NN/TI gap metrics" metrics_path

    return nn_res, ti_res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
