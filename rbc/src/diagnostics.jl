# NN-vs-TI comparison metrics, shared by training, compare.jl, and
# diagnose_divergence.jl.

"""
    gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))

Compare two `simulate` results and return a `Dict{String,Any}` with, per
series: `rmse`, `max_abs`, `nrmse_vs_ti_std` (`rmse / std(ti)`), and
`level_ratio` (`mean(nn) / mean(ti)`); plus an `"aggregate"` entry with
`mean_nrmse` / `max_nrmse` across `series`.
"""
function gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))
    by_var = Dict{String,Any}()
    nrmses = Float64[]
    for s in series
        diff = getfield(nn_res, s) .- getfield(ti_res, s)
        ti = getfield(ti_res, s)
        rmse = sqrt(mean(abs2, diff))
        # population std (corrected=false): scale-free even for short panels
        nrmse = rmse / (std(ti; corrected=false) + 1e-10)
        by_var[string(s)] = Dict(
            "rmse" => rmse,
            "max_abs" => maximum(abs, diff),
            "nrmse_vs_ti_std" => nrmse,
            "level_ratio" => (mean(getfield(nn_res, s)) + 1e-10) / (mean(ti) + 1e-10),
        )
        push!(nrmses, nrmse)
    end
    by_var["aggregate"] = Dict("mean_nrmse" => mean(nrmses), "max_nrmse" => maximum(nrmses))
    return by_var
end
