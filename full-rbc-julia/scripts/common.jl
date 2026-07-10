# Shared helpers for the entry-point scripts: CLI parsing, checkpoint loading,
# path tables, and the Makie comparison figure.
# Expects `src/FullRBC.jl` to have been include()d first.

using .FullRBC
using CSV
using DataFrames
using CairoMakie

const ROOT = dirname(@__DIR__)
const CHECKPOINT_PATH = joinpath(ROOT, "rbc_nn.bson")
const SERIES = (:consumption, :capital, :output, :investment, :productivity)
const SERIES_TITLES = Dict(
    :consumption => "Consumption", :capital => "Capital", :output => "Output",
    :investment => "Investment", :productivity => "TFP (productivity)",
)

"Parse `--flag value` pairs into a Dict; every flag must have a value."
function parse_cli(args)
    d = Dict{String,String}()
    for i in 1:2:length(args)
        startswith(args[i], "--") || error("Expected a flag like --epochs, got: $(args[i])")
        i + 1 <= length(args) || error("Flag $(args[i]) is missing a value")
        d[args[i][3:end]] = args[i+1]
    end
    return d
end

cli_get(kv, key, default::Int) = parse(Int, get(kv, key, string(default)))
cli_get(kv, key, default::Float64) = parse(Float64, get(kv, key, string(default)))
cli_get(kv, key, default::String) = get(kv, key, default)

function load_nn_solver()
    isfile(CHECKPOINT_PATH) ||
        error("No checkpoint at $CHECKPOINT_PATH. Run `julia --project=. -t auto scripts/train.jl` first.")
    return load_checkpoint(CHECKPOINT_PATH)
end

"Write NN/TI/diff columns for every series to a CSV path table."
function save_path_table(nn_res, ti_res, path)
    df = DataFrame(t=0:(length(nn_res.consumption)-1))
    for s in SERIES
        df[!, Symbol(s, :_nn)] = getfield(nn_res, s)
        df[!, Symbol(s, :_ti)] = getfield(ti_res, s)
        df[!, Symbol(s, :_diff)] = getfield(nn_res, s) .- getfield(ti_res, s)
    end
    CSV.write(path, df)
    return path
end

"""
    comparison_figure(nn_res, ti_res; params=nothing) -> Figure

3×2 grid of NN (solid) vs TI (dashed) paths for all series, with a shared
legend. When `params` is given, the free cell carries the calibration.
"""
function comparison_figure(nn_res, ti_res; params::Union{Nothing,RBCParams}=nothing)
    t = 0:(length(nn_res.consumption)-1)
    fig = Figure(size=(1100, 950))
    Label(fig[0, 1:2], "RBC: Neural network vs Time Iteration (same calibration, same shocks)";
          fontsize=18, font=:bold)

    local ax
    for (idx, s) in enumerate(SERIES)
        row, col = fldmod1(idx, 2)
        ax = Axis(fig[row, col], title=SERIES_TITLES[s], xlabel="t")
        lines!(ax, t, getfield(nn_res, s); label="NN", linewidth=2)
        lines!(ax, t, getfield(ti_res, s); label="TI", linewidth=2, linestyle=:dash)
    end

    # SERIES fills cells (1,1)..(3,1); (3,2) is free for legend + calibration.
    # tellwidth/tellheight=false keep this content from shrinking column 2.
    free = GridLayout(fig[3, 2])
    Legend(free[1, 1], ax; orientation=:horizontal, tellwidth=false, tellheight=false)
    if params !== nothing
        lines_txt = join([
            "alpha     = $(round(params.alpha, digits=4))",
            "beta      = $(round(params.beta, digits=4))",
            "delta     = $(round(params.delta, digits=4))",
            "rho       = $(round(params.rho, digits=4))",
            "gamma     = $(round(params.gamma, digits=4))",
            "sigma_eps = $(round(params.sigma_eps, digits=4))",
        ], "\n")
        Label(free[2, 1], "Calibration\n" * lines_txt;
              halign=:center, valign=:top, justification=:left, font="Courier New",
              tellwidth=false, tellheight=false)
    end
    return fig
end
