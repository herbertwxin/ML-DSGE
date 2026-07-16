#=
Single-calibration NN vs TI comparison.

Usage:
    julia --project=. scripts/compare.jl                  # baseline RBC
    julia --project=. scripts/compare.jl --model labor    # RBC with labor choice

Requires the model's trained checkpoint (`rbc_nn.bson` / `rbc_labor_nn.bson`,
override with `--checkpoint`; run `scripts/train.jl --model <name>` first).
The script errors if the checkpoint belongs to a different model and logs the
model actually in use.

Writes, under the project root (labor outputs carry a `rbc_labor_` prefix):
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
trained on (see the params struct's `*_bounds` fields).
"""
get_calibration_params(model::String) =
    model == "rbc" ?
        RBCParams(alpha=0.30, beta=0.98, delta=0.08, rho=0.88, gamma=3.0, sigma_eps=0.02) :
        RBCLaborParams(alpha=0.30, beta=0.98, delta=0.08, rho=0.88, gamma=3.0, sigma_eps=0.02, nu=1.2)

function run_comparison(;
    model::String="rbc",
    params=get_calibration_params(model),
    checkpoint::Union{Nothing,String}=nothing,
)
    spec = model_spec(model)
    prefix = model == "rbc" ? "rbc" : "rbc_$model"

    nn_solver = load_nn_solver(spec; checkpoint)
    ti_policy = solve(spec.benchmark(params); verbose=true)
    ti_policy.converged || @warn "TI benchmark did not converge; metrics may be unreliable"

    nn_res = simulate(nn_solver, params; T=T_SIM, rng=Xoshiro(SIM_SEED))
    ti_res = simulate(ti_policy; T=T_SIM, rng=Xoshiro(SIM_SEED))

    plot_path = joinpath(ROOT, "$(prefix)_comparison.png")
    save(plot_path, comparison_figure(nn_res, ti_res; params, series=spec.series))
    @info "Saved comparison figure" plot_path

    paths_path = save_path_table(nn_res, ti_res, joinpath(ROOT, "$(prefix)_paths.csv");
                                 series=spec.series)
    @info "Saved NN/TI paths" paths_path

    metrics = gap_metrics(nn_res, ti_res; series=spec.series)
    metrics_path = joinpath(ROOT, "$(prefix)_metrics.json")
    open(metrics_path, "w") do io
        JSON.print(io, Dict("model" => spec.name, "params" => params_to_dict(params),
                            "metrics" => metrics), 2)
    end
    @info "Saved NN/TI gap metrics" metrics_path

    return nn_res, ti_res
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = parse_cli(ARGS)
    run_comparison(
        model=cli_get(kv, "model", "rbc"),
        checkpoint=haskey(kv, "checkpoint") ? kv["checkpoint"] : nothing,
    )
end
