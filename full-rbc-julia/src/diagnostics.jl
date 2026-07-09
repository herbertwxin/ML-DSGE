# Shared NN-vs-TI comparison metrics.
# Julia port of `RBCSolver._gap_metrics` (learn_rbc.py) and
# `compute_gap_metrics` (diagnose_divergence.py / compare_rbc.py) — the two
# near-duplicate Python implementations are consolidated into one function here.

"""
    gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))

Compare two simulation results (`NamedTuple`s with the same field names, as
returned by `simulate`) and return a `Dict{String,Any}` with, for each series
in `series`: `rmse`, `max_abs`, `nrmse_vs_ti_std` (`rmse / std(ti)`), and
`level_ratio` (`mean(nn) / mean(ti)`); plus an `"aggregate"` entry with
`mean_nrmse` and `max_nrmse` across `series`.
"""
function gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))
    by_var = Dict{String,Any}()
    nrmses = Float64[]
    for s in series
        nn = getfield(nn_res, s)
        ti = getfield(ti_res, s)
        diff = nn .- ti
        rmse = sqrt(mean(diff .^ 2))
        max_abs = maximum(abs.(diff))
        # corrected=false matches numpy's default population std (ddof=0)
        nrmse = rmse / (std(ti; corrected=false) + 1e-10)
        lvl_ratio = (mean(nn) + 1e-10) / (mean(ti) + 1e-10)
        by_var[string(s)] = Dict(
            "rmse" => rmse, "max_abs" => max_abs,
            "nrmse_vs_ti_std" => nrmse, "level_ratio" => lvl_ratio,
        )
        push!(nrmses, nrmse)
    end
    by_var["aggregate"] = Dict("mean_nrmse" => mean(nrmses), "max_nrmse" => maximum(nrmses))
    return by_var
end
