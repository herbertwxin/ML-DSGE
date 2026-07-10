#=
NN-vs-TI divergence diagnostics over random structural-parameter draws.

Usage:
    julia --project=. -t auto scripts/diagnose_divergence.jl --n-cases 40 --top-k 5 --T 200

Requires a trained checkpoint at `rbc_nn.bson` (run `scripts/train.jl` first).
Cases are evaluated in parallel across Julia threads (`-t auto` / `-t N`).

Outputs under `<output-dir>/` (default `simulation/`, relative to the project
root), with case files ordered by divergence rank:
    divergence_summary.csv
    divergence_top_cases.json
    divergence_case_XXX_paths.csv
    divergence_case_XXX_plot.png
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
include(joinpath(@__DIR__, "common.jl"))
using .FullRBC
using Random
using Statistics
using JSON

"Fraction of a path outside `[lo, hi]`."
oob_frac(x, lo, hi) = mean((x .< lo) .| (x .> hi))

function run_case(nn_solver::NNSolver, params::RBCParams, T::Int, seed::Int)
    ti_policy = solve(TISolver(params))
    nn_res = simulate(nn_solver, params; T, rng=Xoshiro(seed))
    ti_res = simulate(ti_policy; T, rng=Xoshiro(seed))

    metrics = gap_metrics(nn_res, ti_res; series=SERIES)
    k_low, k_high = k_support(params)
    a_low, a_high = a_support_from_shock_params(params.rho, params.sigma_eps, params.A_sigma_mult)
    diagnostics = Dict(
        "ti_converged" => ti_policy.converged,
        "nn_k_oob_frac" => oob_frac(nn_res.capital, k_low, k_high),
        "nn_a_oob_frac" => oob_frac(nn_res.productivity, a_low, a_high),
        "ti_k_oob_frac" => oob_frac(ti_res.capital, ti_policy.k_min, ti_policy.k_max),
        "ti_a_oob_frac" => oob_frac(ti_res.productivity, ti_policy.A_min, ti_policy.A_max),
    )

    summary = Dict{String,Any}(
        "score_max_nrmse" => metrics["aggregate"]["max_nrmse"],
        "mean_nrmse" => metrics["aggregate"]["mean_nrmse"],
        "nrmse_c" => metrics["consumption"]["nrmse_vs_ti_std"],
        "nrmse_k" => metrics["capital"]["nrmse_vs_ti_std"],
        "nrmse_y" => metrics["output"]["nrmse_vs_ti_std"],
        "nrmse_i" => metrics["investment"]["nrmse_vs_ti_std"],
        "nrmse_A" => metrics["productivity"]["nrmse_vs_ti_std"],
    )
    for f in (:alpha, :beta, :delta, :rho, :gamma, :sigma_eps)
        summary[string(f)] = getfield(params, f)
    end
    merge!(summary, diagnostics)

    detail = Dict(
        "params" => params_to_dict(params),
        "metrics" => metrics,
        "diagnostics" => diagnostics,
    )
    return summary, detail, (nn=nn_res, ti=ti_res)
end

function write_summary_csv(summaries, order, path)
    cols = ["score_max_nrmse", "alpha", "beta", "delta", "rho", "gamma", "sigma_eps",
            "ti_converged", "ti_k_oob_frac", "ti_a_oob_frac", "nn_k_oob_frac", "nn_a_oob_frac",
            "mean_nrmse", "nrmse_c", "nrmse_k", "nrmse_y", "nrmse_i", "nrmse_A"]
    df = DataFrame([col => [summaries[idx][col] for idx in order] for col in cols])
    CSV.write(path, df)
    return path
end

function main(; n_cases::Int=40, top_k::Int=5, T::Int=200, seed::Int=123, output_dir::String="simulation")
    outdir = joinpath(ROOT, output_dir)
    mkpath(outdir)
    @info "Saving simulation outputs" outdir

    nn_solver = load_nn_solver()
    rng = Xoshiro(seed)
    sampled_params = [sample_params_uniform(RBCParams(), rng) for _ in 1:n_cases]

    summaries = Vector{Any}(undef, n_cases)
    details = Vector{Any}(undef, n_cases)
    paths_cache = Vector{Any}(undef, n_cases)

    @info "Running $n_cases cases across $(Threads.nthreads()) thread(s)..."
    Threads.@threads for i in 1:n_cases
        s, d, paths = run_case(nn_solver, sampled_params[i], T, seed + i)
        summaries[i], details[i], paths_cache[i] = s, d, paths
        @info "case done" i n_cases score = s["score_max_nrmse"] beta = s["beta"] delta = s["delta"] rho = s["rho"]
    end

    order = sortperm([s["score_max_nrmse"] for s in summaries]; rev=true)
    top_k = min(top_k, n_cases)

    # Paths and figures for every case, ordered by divergence rank.
    # (CairoMakie rendering is kept out of the threaded loop.)
    for (rank, idx) in enumerate(order)
        tag = lpad(rank, 3, '0')
        save_path_table(paths_cache[idx].nn, paths_cache[idx].ti,
                        joinpath(outdir, "divergence_case_$(tag)_paths.csv"))
        fig = comparison_figure(paths_cache[idx].nn, paths_cache[idx].ti; params=sampled_params[idx])
        save(joinpath(outdir, "divergence_case_$(tag)_plot.png"), fig)
    end

    summary_path = write_summary_csv(summaries, order, joinpath(outdir, "divergence_summary.csv"))
    @info "Saved case summary" summary_path

    top_cases = [
        Dict(
            "rank" => rank,
            "summary" => summaries[idx],
            "detail" => details[idx],
            "paths_file" => joinpath(outdir, "divergence_case_$(lpad(rank, 3, '0'))_paths.csv"),
            "plot_file" => joinpath(outdir, "divergence_case_$(lpad(rank, 3, '0'))_plot.png"),
        )
        for (rank, idx) in enumerate(order[1:top_k])
    ]
    top_path = joinpath(outdir, "divergence_top_cases.json")
    open(top_path, "w") do io
        JSON.print(io, top_cases, 2)
    end
    @info "Saved top-case diagnostics" top_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = parse_cli(ARGS)
    main(
        n_cases=cli_get(kv, "n-cases", 40),
        top_k=cli_get(kv, "top-k", 5),
        T=cli_get(kv, "T", 200),
        seed=cli_get(kv, "seed", 123),
        output_dir=cli_get(kv, "output-dir", "simulation"),
    )
end
