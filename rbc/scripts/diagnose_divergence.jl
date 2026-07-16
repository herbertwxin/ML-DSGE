#=
NN-vs-TI divergence diagnostics over random structural-parameter draws.

Usage:
    julia --project=. -t auto scripts/diagnose_divergence.jl --n-cases 40 --top-k 5 --T 200
    julia --project=. -t auto scripts/diagnose_divergence.jl --model labor --n-cases 40

`--model rbc|labor` picks the model (default rbc): its checkpoint
(`rbc_nn.bson` / `rbc_labor_nn.bson`, override with `--checkpoint`), its TI
benchmark, and its series/parameter columns. The script errors if the
checkpoint on disk belongs to a different model and logs the model actually
in use. Cases are evaluated in parallel across Julia threads (`-t auto`).

Outputs under `<output-dir>/` (default `simulation/` for rbc,
`simulation_labor/` for labor; relative to the project root), with case
files ordered by divergence rank:
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

function run_case(spec, nn_solver::NNSolver, params, T::Int, seed::Int)
    ti_policy = solve(spec.benchmark(params))
    nn_res = simulate(nn_solver, params; T, rng=Xoshiro(seed))
    ti_res = simulate(ti_policy; T, rng=Xoshiro(seed))

    metrics = gap_metrics(nn_res, ti_res; series=spec.series)
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
    )
    for s in spec.series
        summary["nrmse_$(SERIES_SHORT[s])"] = metrics[string(s)]["nrmse_vs_ti_std"]
    end
    for f in spec.structural
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

function write_summary_csv(spec, summaries, order, path)
    cols = ["score_max_nrmse"; string.(collect(spec.structural));
            "ti_converged"; "ti_k_oob_frac"; "ti_a_oob_frac"; "nn_k_oob_frac"; "nn_a_oob_frac";
            "mean_nrmse"; ["nrmse_$(SERIES_SHORT[s])" for s in spec.series]]
    df = DataFrame([col => [summaries[idx][col] for idx in order] for col in cols])
    CSV.write(path, df)
    return path
end

function main(; model::String="rbc", n_cases::Int=40, top_k::Int=5, T::Int=200, seed::Int=123,
              checkpoint::Union{Nothing,String}=nothing, output_dir::Union{Nothing,String}=nothing)
    spec = model_spec(model)
    output_dir = something(output_dir, model == "rbc" ? "simulation" : "simulation_$model")
    outdir = joinpath(ROOT, output_dir)
    mkpath(outdir)
    @info "Divergence sweep" model = spec.name n_cases T outdir

    nn_solver = load_nn_solver(spec; checkpoint)
    rng = Xoshiro(seed)
    sampled_params = [sample_params_uniform(spec.params(), rng) for _ in 1:n_cases]

    summaries = Vector{Any}(undef, n_cases)
    details = Vector{Any}(undef, n_cases)
    paths_cache = Vector{Any}(undef, n_cases)

    @info "Running $n_cases cases across $(Threads.nthreads()) thread(s)..."
    Threads.@threads for i in 1:n_cases
        s, d, paths = run_case(spec, nn_solver, sampled_params[i], T, seed + i)
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
                        joinpath(outdir, "divergence_case_$(tag)_paths.csv"); series=spec.series)
        fig = comparison_figure(paths_cache[idx].nn, paths_cache[idx].ti;
                                params=sampled_params[idx], series=spec.series)
        save(joinpath(outdir, "divergence_case_$(tag)_plot.png"), fig)
    end

    summary_path = write_summary_csv(spec, summaries, order, joinpath(outdir, "divergence_summary.csv"))
    @info "Saved case summary" summary_path

    top_cases = [
        Dict(
            "rank" => rank,
            "model" => spec.name,
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
        model=cli_get(kv, "model", "rbc"),
        n_cases=cli_get(kv, "n-cases", 40),
        top_k=cli_get(kv, "top-k", 5),
        T=cli_get(kv, "T", 200),
        seed=cli_get(kv, "seed", 123),
        checkpoint=haskey(kv, "checkpoint") ? kv["checkpoint"] : nothing,
        output_dir=haskey(kv, "output-dir") ? kv["output-dir"] : nothing,
    )
end
