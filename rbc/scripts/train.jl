#=
Train a neural-network policy for one of the models.

Usage:
    julia --project=. -t auto scripts/train.jl                     # baseline RBC
    julia --project=. -t auto scripts/train.jl --model labor       # RBC with labor choice
    julia --project=. -t auto scripts/train.jl --epochs 20000 --batch-size 1024
    julia --project=. -t auto scripts/train.jl --device gpu        # or cpu; default auto

Writes, under the project root:
    rbc_nn.bson / rbc_labor_nn.bson   trained checkpoint (weights + params struct)
    learn_<model>_loss.png            training loss curve
=#
include(joinpath(@__DIR__, "..", "src", "FullRBC.jl"))
include(joinpath(@__DIR__, "common.jl"))
using .FullRBC
using Random

function save_loss_plot(losses, path, model_name)
    fig = Figure(size=(850, 550))
    ax = Axis(fig[1, 1]; yscale=log10, xlabel="Epoch",
              ylabel="Train loss (residual MSE + penalties)",
              title="NN training loss ($model_name)")
    lines!(ax, losses; linewidth=1)
    save(path, fig)
    return path
end

function main(;
    model::String="rbc",                      # "rbc" or "labor"
    batch_size::Union{Int,Nothing}=nothing,   # default: 2048 CPU / 32768 GPU
    epochs::Int=50_000,
    eval_every::Int=200,
    val_batch_size::Int=8192,
    patience::Int=20,
    min_rel_improve::Float64=5e-3,
    k_oob_weight::Float64=1.0,
    intra_weight::Float64=1.0,                # labor model only
    panel_n_cases::Int=4,
    panel_T::Int=120,
    panel_seed::Int=321,
    seed::Int=42,
    checkpoint_path::Union{Nothing,String}=nothing,
    device::String="auto",
)
    params = model == "rbc"   ? RBCParams() :
             model == "labor" ? RBCLaborParams() :
             error("Unknown --model '$model' (use rbc or labor)")
    loss_kwargs = model == "labor" ? (; k_oob_weight, intra_weight) : (; k_oob_weight)
    checkpoint_path = something(checkpoint_path,
        model == "rbc" ? CHECKPOINT_PATH : joinpath(ROOT, "rbc_labor_nn.bson"))

    solver = NNSolver(params; device=select_device(Symbol(device)))
    batch_size = something(batch_size, default_batch_size(solver.device))
    @info "Batch size" batch_size
    losses = train!(
        solver;
        batch_size, epochs, eval_every, val_batch_size, patience, min_rel_improve,
        loss_kwargs, panel_n_cases, panel_T, panel_seed,
        best_checkpoint_path=checkpoint_path, rng=Xoshiro(seed),
    )

    # solver.model holds the best-validation weights after train!; save once
    # more so the checkpoint on disk matches the returned solver.
    save_checkpoint(solver, checkpoint_path)
    @info "Saved trained checkpoint" checkpoint_path

    loss_plot_path = save_loss_plot(
        losses, joinpath(dirname(checkpoint_path), "learn_$(model)_loss.png"), model)
    @info "Saved loss plot" loss_plot_path

    return solver, losses
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = parse_cli(ARGS)
    main(
        model=cli_get(kv, "model", "rbc"),
        batch_size=haskey(kv, "batch-size") ? parse(Int, kv["batch-size"]) : nothing,
        epochs=cli_get(kv, "epochs", 50_000),
        eval_every=cli_get(kv, "eval-every", 200),
        val_batch_size=cli_get(kv, "val-batch-size", 8192),
        patience=cli_get(kv, "patience", 20),
        min_rel_improve=cli_get(kv, "min-rel-improve", 5e-3),
        k_oob_weight=cli_get(kv, "k-oob-weight", 1.0),
        intra_weight=cli_get(kv, "intra-weight", 1.0),
        panel_n_cases=cli_get(kv, "panel-n-cases", 4),
        panel_T=cli_get(kv, "panel-T", 120),
        panel_seed=cli_get(kv, "panel-seed", 321),
        seed=cli_get(kv, "seed", 42),
        checkpoint_path=haskey(kv, "checkpoint") ? kv["checkpoint"] : nothing,
        device=cli_get(kv, "device", "auto"),
    )
end
