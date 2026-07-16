# Shared helpers for the entry-point scripts: the model registry, CLI parsing,
# checkpoint loading, path tables, and the Makie comparison figure.
# Expects `src/FullRBC.jl` to have been include()d first.

using .FullRBC
using CSV
using DataFrames
using CairoMakie

const ROOT = dirname(@__DIR__)

# ---------------------------------------------------------------------------
# Model registry: everything the scripts need to know about a model, in one
# place. Adding a model to the scripts = adding an entry here.
# ---------------------------------------------------------------------------

const MODELS = Dict(
    "rbc" => (
        name="rbc",
        params=RBCParams,
        benchmark=TISolver,
        default_checkpoint="rbc_nn.bson",
        series=(:consumption, :capital, :output, :investment, :productivity),
        structural=(:alpha, :beta, :delta, :rho, :gamma, :sigma_eps),
    ),
    "labor" => (
        name="labor",
        params=RBCLaborParams,
        benchmark=LaborTISolver,
        default_checkpoint="rbc_labor_nn.bson",
        series=(:consumption, :capital, :output, :investment, :productivity, :hours),
        structural=(:alpha, :beta, :delta, :rho, :gamma, :sigma_eps, :nu),
    ),
)

model_spec(name::String) = haskey(MODELS, name) ? MODELS[name] :
    error("Unknown --model '$name' (choose from: $(join(sort(collect(keys(MODELS))), ", ")))")

"Registry entry for the model a params struct (or solver) belongs to."
function model_spec_for(p)
    for spec in values(MODELS)
        p isa spec.params && return spec
    end
    error("No registered model for $(typeof(p))")
end

const SERIES = MODELS["rbc"].series   # baseline series (default for helpers)
const SERIES_TITLES = Dict(
    :consumption => "Consumption", :capital => "Capital", :output => "Output",
    :investment => "Investment", :productivity => "TFP (productivity)",
    :hours => "Hours worked",
)
# Short column suffixes for sweep CSVs (nrmse_c, nrmse_k, ...).
const SERIES_SHORT = Dict(
    :consumption => "c", :capital => "k", :output => "y",
    :investment => "i", :productivity => "A", :hours => "n",
)

# ---------------------------------------------------------------------------
# CLI and checkpoint loading
# ---------------------------------------------------------------------------

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

"""
    load_nn_solver(spec; checkpoint=nothing)

Load the NN checkpoint for the model in registry entry `spec` (default path
`ROOT/spec.default_checkpoint`, overridable). Errors if the checkpoint's
stored parameter struct belongs to a different model than requested, so a
sweep or comparison can never silently run the wrong model; logs the model
name, parameter type, and path that are actually in use.
"""
function load_nn_solver(spec; checkpoint::Union{Nothing,String}=nothing)
    path = something(checkpoint, joinpath(ROOT, spec.default_checkpoint))
    isfile(path) || error("No checkpoint at $path. Train it first: " *
        "`julia --project=. -t auto scripts/train.jl --model $(spec.name)`.")
    solver = load_checkpoint(path)
    solver.p isa spec.params || error(
        "Checkpoint at $path holds $(typeof(solver.p)), but --model $(spec.name) " *
        "expects $(spec.params). Pass a matching --model or --checkpoint.")
    @info "NN solver ready" model = spec.name params_type = typeof(solver.p) checkpoint = path
    return solver
end

# ---------------------------------------------------------------------------
# Output helpers (series-generic)
# ---------------------------------------------------------------------------

"Write NN/TI/diff columns for every series to a CSV path table."
function save_path_table(nn_res, ti_res, path; series=SERIES)
    df = DataFrame(t=0:(length(nn_res.consumption)-1))
    for s in series
        df[!, Symbol(s, :_nn)] = getfield(nn_res, s)
        df[!, Symbol(s, :_ti)] = getfield(ti_res, s)
        df[!, Symbol(s, :_diff)] = getfield(nn_res, s) .- getfield(ti_res, s)
    end
    CSV.write(path, df)
    return path
end

"""
    comparison_figure(nn_res, ti_res; params=nothing, series=SERIES) -> Figure

Two-column grid of NN (solid) vs TI (dashed) paths for `series`, with a
shared legend in the first free cell. When `params` is given, the free cell
also carries the calibration (structural fields from the model registry).
"""
function comparison_figure(nn_res, ti_res; params=nothing, series=SERIES)
    t = 0:(length(nn_res.consumption)-1)
    n_cells = length(series) + 1                 # series panels + legend cell
    n_rows = cld(n_cells, 2)
    fig = Figure(size=(1100, 300 * n_rows + 50))
    Label(fig[0, 1:2], "Neural network vs Time Iteration (same calibration, same shocks)";
          fontsize=18, font=:bold)

    local ax
    for (idx, s) in enumerate(series)
        row, col = fldmod1(idx, 2)
        ax = Axis(fig[row, col], title=SERIES_TITLES[s], xlabel="t")
        lines!(ax, t, getfield(nn_res, s); label="NN", linewidth=2)
        lines!(ax, t, getfield(ti_res, s); label="TI", linewidth=2, linestyle=:dash)
    end

    # First cell after the series panels: legend + calibration.
    # tellwidth/tellheight=false keep this content from shrinking its column.
    free = GridLayout(fig[fldmod1(n_cells, 2)...])
    Legend(free[1, 1], ax; orientation=:horizontal, tellwidth=false, tellheight=false)
    if params !== nothing
        fields = model_spec_for(params).structural
        lines_txt = join(
            ["$(rpad(string(f), 9)) = $(round(getfield(params, f), digits=4))" for f in fields],
            "\n")
        Label(free[2, 1], "Calibration ($(nameof(typeof(params))))\n" * lines_txt;
              halign=:center, valign=:top, justification=:left, font="Courier New",
              tellwidth=false, tellheight=false)
    end
    return fig
end
