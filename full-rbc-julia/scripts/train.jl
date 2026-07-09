#=
Canonical training entrypoint for the RBC NN (Julia port of `learn_rbc.py`'s
`train_rbc_model` / `train_rbc_cli`).

Usage:
    julia --project=. -t auto scripts/train.jl
    julia --project=. -t auto scripts/train.jl --epochs 20000 --batch-size 1024

Writes, under the project root (one level up from `scripts/`):
    rbc_nn.bson         trained checkpoint (model weights + RBCParams)
    learn_rbc_loss.png  training loss curve
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
using .FullRBC
using Random
using Plots

const ROOT = dirname(@__DIR__)

function parse_cli(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        @assert startswith(key, "--") "Expected a flag like --epochs, got: $key"
        d[key[3:end]] = args[i+1]
        i += 2
    end
    return d
end

function main(;
    batch_size::Int=2048,
    epochs::Int=50000,
    eval_every::Int=200,
    val_batch_size::Int=8192,
    patience::Int=20,
    min_rel_improve::Float64=5e-3,
    panel_n_cases::Int=4,
    panel_T::Int=120,
    panel_seed::Int=321,
    seed::Int=42,
)
    checkpoint_path = joinpath(ROOT, "rbc_nn.bson")
    loss_plot_path = joinpath(ROOT, "learn_rbc_loss.png")

    p = RBCParams()
    solver = NNSolver(p)
    rng = Xoshiro(seed)

    losses = train!(
        solver;
        batch_size=batch_size, epochs=epochs, eval_every=eval_every,
        val_batch_size=val_batch_size, patience=patience, min_rel_improve=min_rel_improve,
        panel_n_cases=panel_n_cases, panel_T=panel_T, panel_seed=panel_seed,
        best_checkpoint_path=checkpoint_path, rng=rng,
    )

    # solver.model already holds the best-validation weights after train!; save once more
    # so the checkpoint on disk always reflects the returned solver.
    save_checkpoint(solver, checkpoint_path)
    @info "Saved trained checkpoint" checkpoint_path

    plt = plot(losses; yscale=:log10, xlabel="Epoch", ylabel="Train MSE (normalized Euler residual)",
               title="RBC NN training loss", legend=false)
    savefig(plt, loss_plot_path)
    @info "Saved loss plot" loss_plot_path

    return solver, losses
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = parse_cli(ARGS)
    main(
        batch_size=parse(Int, get(kv, "batch-size", "2048")),
        epochs=parse(Int, get(kv, "epochs", "50000")),
        eval_every=parse(Int, get(kv, "eval-every", "200")),
        val_batch_size=parse(Int, get(kv, "val-batch-size", "8192")),
        patience=parse(Int, get(kv, "patience", "20")),
        min_rel_improve=parse(Float64, get(kv, "min-rel-improve", "5e-3")),
        panel_n_cases=parse(Int, get(kv, "panel-n-cases", "4")),
        panel_T=parse(Int, get(kv, "panel-T", "120")),
        panel_seed=parse(Int, get(kv, "panel-seed", "321")),
        seed=parse(Int, get(kv, "seed", "42")),
    )
end
