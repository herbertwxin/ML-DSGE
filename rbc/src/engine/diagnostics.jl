# Solver-vs-benchmark comparison metrics. Model-agnostic: works on any two
# `simulate` results with matching named fields.

"""
    gap_metrics(nn_res, ti_res; series=propertynames(ti_res))

Compare two `simulate` results and return a `Dict{String,Any}` with, per
series: `rmse`, `max_abs`, `nrmse_vs_ti_std` (`rmse / std(ti)`), and
`level_ratio` (`mean(nn) / mean(ti)`); plus an `"aggregate"` entry with
`mean_nrmse` / `max_nrmse` across `series`. By default all series in
`ti_res` are compared; pass `series` to restrict.
"""
function gap_metrics(nn_res, ti_res; series=propertynames(ti_res))
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
