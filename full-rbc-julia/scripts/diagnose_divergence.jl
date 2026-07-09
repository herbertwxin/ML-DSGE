#=
Automatic NN-vs-TI divergence diagnostics over random parameter draws (Julia
port of `diagnose_divergence.py`).

Usage:
    julia --project=. -t auto scripts/diagnose_divergence.jl --n-cases 40 --top-k 5 --T 200

Requires a trained checkpoint at `rbc_nn.bson` (run `scripts/train.jl` first).
Cases are evaluated in parallel across Julia threads (`-t auto` / `-t N`),
replacing the Python version's `--n-workers` multiprocessing flag.

Outputs under `<output-dir>/` (default `simulation/`, resolved relative to the
project root):
    divergence_summary.csv
    divergence_top_cases.json
    divergence_case_XXX_paths.csv
    divergence_case_XXX_plot.png
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
using .FullRBC
using Random
using Statistics
using CSV
using DataFrames
using JSON
using Plots

const ROOT = dirname(@__DIR__)
const CHECKPOINT_PATH = joinpath(ROOT, "rbc_nn.bson")
const SERIES = (:consumption, :capital, :output, :investment, :productivity)

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
end

function save_comparison_plot(nn_res, ti_res, path, params::RBCParams)
    t = 0:(length(nn_res.consumption)-1)
    titles = Dict(:consumption => "Consumption", :capital => "Capital", :output => "Output",
                  :investment => "Investment", :productivity => "TFP (productivity)")
    subplots = [
        plot(t, [getfield(nn_res, s) getfield(ti_res, s)], label=["NN" "TI"],
             linestyle=[:solid :dash], title=titles[s], xlabel="t")
        for s in SERIES
    ]
    param_text = """
    Parameter set
    alpha        = $(round(params.alpha, digits=4))
    beta         = $(round(params.beta, digits=4))
    delta        = $(round(params.delta, digits=4))
    rho          = $(round(params.rho, digits=4))
    gamma        = $(round(params.gamma, digits=4))
    sigma_eps    = $(round(params.sigma_eps, digits=4))
    A_sigma_mult = $(round(params.A_sigma_mult, digits=2))"""
    text_panel = plot(; framestyle=:none, legend=false, xlim=(0, 1), ylim=(0, 1))
    annotate!(text_panel, 0.0, 1.0, text(param_text, 9, :left, :top))

    fig = plot(subplots..., text_panel, layout=(3, 2), size=(1000, 1000),
               plot_title="RBC: Neural network vs Time Iteration (same calibration, same seed)")
    savefig(fig, path)
end

function run_case(nn_solver::NNSolver, params::RBCParams, T::Int, seed::Int)
    ti = TISolver(params)
    policy = solve!(ti)

    nn_res = simulate(nn_solver; T=T, alpha=params.alpha, beta=params.beta, delta=params.delta,
                       rho=params.rho, gamma=params.gamma, sigma_eps=params.sigma_eps, rng=Xoshiro(seed))
    ti_res = simulate(ti, policy; T=T, rng=Xoshiro(seed))

    metrics = gap_metrics(nn_res, ti_res; series=SERIES)
    score = metrics["aggregate"]["max_nrmse"]

    k_ss_sim, _, _ = steady_state_batch(params.alpha, params.beta, params.delta)
    nn_k_low = params.k_bounds[1] * k_ss_sim
    nn_k_high = params.k_bounds[2] * k_ss_sim
    nn_k_oob = mean((nn_res.capital .< nn_k_low) .| (nn_res.capital .> nn_k_high))

    a_low, a_high = a_support_from_shock_params(params.rho, params.sigma_eps, params.A_sigma_mult, 1.0)
    nn_a_oob = mean((nn_res.productivity .< a_low) .| (nn_res.productivity .> a_high))

    ti_k_oob = mean((ti_res.capital .< ti.k_min) .| (ti_res.capital .> ti.k_max))
    ti_a_oob = mean((ti_res.productivity .< ti.A_min) .| (ti_res.productivity .> ti.A_max))

    summary = Dict(
        "score_max_nrmse" => score, "alpha" => params.alpha, "beta" => params.beta, "delta" => params.delta,
        "rho" => params.rho, "gamma" => params.gamma, "sigma_eps" => params.sigma_eps,
        "ti_k_oob_frac" => ti_k_oob, "ti_a_oob_frac" => ti_a_oob,
        "nn_k_oob_frac" => nn_k_oob, "nn_a_oob_frac" => nn_a_oob,
        "mean_nrmse" => metrics["aggregate"]["mean_nrmse"],
        "nrmse_c" => metrics["consumption"]["nrmse_vs_ti_std"],
        "nrmse_k" => metrics["capital"]["nrmse_vs_ti_std"],
        "nrmse_y" => metrics["output"]["nrmse_vs_ti_std"],
        "nrmse_i" => metrics["investment"]["nrmse_vs_ti_std"],
        "nrmse_A" => metrics["productivity"]["nrmse_vs_ti_std"],
    )
    detail = Dict(
        "params" => params_to_dict(params),
        "metrics" => metrics,
        "diagnostics" => Dict("ti_k_oob_frac" => ti_k_oob, "ti_a_oob_frac" => ti_a_oob,
                               "nn_k_oob_frac" => nn_k_oob, "nn_a_oob_frac" => nn_a_oob),
    )
    return summary, detail, (nn=nn_res, ti=ti_res)
end

function write_summary_csv(summaries, order, path)
    cols = ["score_max_nrmse", "alpha", "beta", "delta", "rho", "gamma", "sigma_eps",
            "ti_k_oob_frac", "ti_a_oob_frac", "nn_k_oob_frac", "nn_a_oob_frac",
            "mean_nrmse", "nrmse_c", "nrmse_k", "nrmse_y", "nrmse_i", "nrmse_A"]
    df = DataFrame([col => [summaries[idx][col] for idx in order] for col in cols])
    CSV.write(path, df)
end

function main(; n_cases::Int=40, top_k::Int=5, T::Int=200, seed::Int=123, output_dir::String="simulation")
    outdir = joinpath(ROOT, output_dir)
    mkpath(outdir)
    @info "Saving simulation outputs to $outdir"

    nn_solver = get_nn_solver()
    base_params = RBCParams()
    rng = Xoshiro(seed)
    sampled_params = [sample_params_uniform(base_params, rng) for _ in 1:n_cases]

    summaries = Vector{Any}(undef, n_cases)
    details = Vector{Any}(undef, n_cases)
    paths_cache = Vector{Any}(undef, n_cases)

    @info "Running $n_cases cases across $(Threads.nthreads()) thread(s)..."
    Threads.@threads for i in 1:n_cases
        s, d, paths = run_case(nn_solver, sampled_params[i], T, seed + i)
        summaries[i] = s
        details[i] = d
        paths_cache[i] = paths
        p_i = sampled_params[i]
        @info "case done" i n_cases score = s["score_max_nrmse"] alpha = p_i.alpha beta = p_i.beta delta = p_i.delta rho = p_i.rho gamma = p_i.gamma sigma_eps = p_i.sigma_eps
    end

    order = sortperm([summaries[i]["score_max_nrmse"] for i in 1:n_cases]; rev=true)
    top_k = min(top_k, length(order))

    # Save full paths and plots for ALL evaluated cases, ordered by divergence rank.
    for (rank, idx) in enumerate(order)
        tag = lpad(rank, 3, '0')
        save_path_table(paths_cache[idx].nn, paths_cache[idx].ti, joinpath(outdir, "divergence_case_$(tag)_paths.csv"))
        save_comparison_plot(paths_cache[idx].nn, paths_cache[idx].ti, joinpath(outdir, "divergence_case_$(tag)_plot.png"), sampled_params[idx])
    end

    summary_path = joinpath(outdir, "divergence_summary.csv")
    write_summary_csv(summaries, order, summary_path)
    @info "Saved case summary" summary_path

    top_cases = []
    for (rank, idx) in enumerate(order[1:top_k])
        tag = lpad(rank, 3, '0')
        push!(top_cases, Dict(
            "rank" => rank,
            "summary" => summaries[idx],
            "detail" => details[idx],
            "paths_file" => joinpath(outdir, "divergence_case_$(tag)_paths.csv"),
            "plot_file" => joinpath(outdir, "divergence_case_$(tag)_plot.png"),
        ))
    end
    top_path = joinpath(outdir, "divergence_top_cases.json")
    open(top_path, "w") do io
        JSON.print(io, top_cases, 2)
    end
    @info "Saved top-case diagnostics" top_path
end

function parse_cli(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        @assert startswith(key, "--") "Expected a flag like --n-cases, got: $key"
        d[key[3:end]] = args[i+1]
        i += 2
    end
    return d
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = parse_cli(ARGS)
    main(
        n_cases=parse(Int, get(kv, "n-cases", "40")),
        top_k=parse(Int, get(kv, "top-k", "5")),
        T=parse(Int, get(kv, "T", "200")),
        seed=parse(Int, get(kv, "seed", "123")),
        output_dir=get(kv, "output-dir", "simulation"),
    )
end
