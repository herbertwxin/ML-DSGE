# Model-agnostic training loop and checkpoint persistence. The model enters
# only through the interface functions (see interface.jl) dispatched on the
# type of `solver.p`.

"""
    train!(solver::NNSolver; batch_size=2048, epochs=50_000, eval_every=200,
           val_batch_size=8192, patience=20, min_rel_improve=5e-3,
           loss_kwargs=(;), panel_n_cases=4, panel_T=120, panel_seed=321,
           best_checkpoint_path=nothing, rng=Random.default_rng())

Minimize the model's [`training_loss`](@ref) over freshly sampled batches,
with early stopping on `validation_report(...).loss` evaluated on a fixed
validation batch. `loss_kwargs` (e.g. penalty weights) are forwarded verbatim
to [`training_loss`](@ref) / [`validation_report`](@ref) — the engine never
interprets them. Every `eval_every` epochs the model's benchmark panel, if it
defines one, is re-evaluated and logged; the panel is *diagnostic only* and
plays no role in stopping or model selection. Returns the per-epoch training
losses; the solver's model holds the best-validation weights on exit.
"""
function train!(solver::NNSolver;
                batch_size::Int=2048, epochs::Int=50_000, eval_every::Int=200,
                val_batch_size::Int=8192, patience::Int=20, min_rel_improve::Float64=5e-3,
                loss_kwargs::NamedTuple=(;),
                panel_n_cases::Int=4, panel_T::Int=120, panel_seed::Int=321,
                best_checkpoint_path::Union{Nothing,String}=nothing,
                rng::AbstractRNG=Random.default_rng())
    p = solver.p
    losses = Float64[]
    @info "Training device" solver.device
    val_batch = solver.device(sample_batch(p, rng, val_batch_size))
    panel = build_validation_panel(p, panel_n_cases, panel_seed)
    isempty(panel) || @info "Validation panel ready" n_cases = length(panel) panel_T

    best_val_loss = Inf
    best_epoch = 0
    bad_evals = 0
    best_state = nothing

    @info "Training" params = typeof(p) loss_kwargs
    for epoch in 1:epochs
        batch = solver.device(sample_batch(p, rng, batch_size))
        loss_val, grads = Flux.withgradient(
            m -> training_loss(p, m, batch, solver.quad; loss_kwargs...), solver.model)
        Flux.update!(solver.opt_state, solver.model, grads[1])
        push!(losses, loss_val)

        epoch % eval_every == 0 || continue

        report = validation_report(p, solver.model, val_batch, solver.quad; loss_kwargs...)
        val_loss = report.loss
        panel_metrics = evaluate_validation_panel(p, solver, panel, panel_T)
        if panel_metrics === nothing
            @info "epoch" epoch train_loss = loss_val validation = report
        else
            @info "epoch" epoch train_loss = loss_val validation = report panel = panel_metrics
        end

        rel_improve = (best_val_loss - val_loss) / max(abs(best_val_loss), 1e-12)
        if val_loss < best_val_loss && (isinf(best_val_loss) || rel_improve >= min_rel_improve)
            best_val_loss = val_loss
            best_epoch = epoch
            bad_evals = 0
            best_state = deepcopy(Flux.state(solver.model))
            best_checkpoint_path === nothing || save_checkpoint(solver, best_checkpoint_path)
        else
            bad_evals += 1
            if bad_evals >= patience
                @info "Early stopping" epoch best_epoch best_val_loss
                break
            end
        end
    end

    if best_state === nothing
        @warn "No best validation checkpoint captured; keeping last-epoch weights."
    else
        Flux.loadmodel!(solver.model, best_state)
        @info "Restored best model" best_epoch best_val_loss
    end
    return losses
end

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

"""
    save_checkpoint(solver::NNSolver, path)

Save `Flux.state(model)` plus the model's parameter struct to a BSON file.
The state is always written as CPU arrays (Float32), so checkpoints are
interchangeable across devices; older Float64 checkpoints load transparently.
"""
function save_checkpoint(solver::NNSolver, path::AbstractString)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    model_state = Flux.state(cpu(solver.model))
    p = solver.p
    BSON.@save path model_state p
    return path
end

"""
    load_checkpoint(path; device=cpu_device()) -> NNSolver

Restore an [`NNSolver`](@ref) saved by [`save_checkpoint`](@ref); the stored
parameter struct picks the model. Defaults to the CPU — the checkpoint's
consumers (`simulate`, `compare.jl`, `diagnose_divergence.jl`) run pointwise
where a GPU would only add transfer overhead; pass `device=select_device()`
to continue *training* on a GPU.
"""
function load_checkpoint(path::AbstractString; device=cpu_device())
    data = BSON.load(path, @__MODULE__)
    solver = NNSolver(data[:p]; device, model_state=data[:model_state])
    @info "Loaded checkpoint" path solver.device
    return solver
end
