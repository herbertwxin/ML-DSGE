# The model interface.
#
# The engine (the other files in this directory) is model-agnostic: it
# reaches the economics only through the generic functions declared here,
# dispatched on the type of the model's parameter struct. To add a model,
# define a parameter struct `P` (structural scalars + training bounds; see
# `RBCParams` for the pattern) and implement, in the model's own file:
#
#     policy_spec(p::P)                                  required
#     sample_batch(p::P, rng, batch_size)                required
#     training_loss(p::P, model, batch, quad; kw...)     required
#     validation_report(p::P, model, batch, quad; kw...) optional
#     build_validation_panel(p::P, n_cases, seed)        optional
#     evaluate_validation_panel(p::P, solver, panel, T)  optional
#
# Loss hyperparameters (e.g. penalty weights) are passed to `train!` as
# `loss_kwargs::NamedTuple` and forwarded verbatim to `training_loss` and
# `validation_report`; the engine never interprets them. Swapping the loss
# therefore only ever touches the model file.

"""
    policy_spec(p) -> (; input_dim, hidden, output_dim, output_bias)

Network architecture for the model behind parameter struct `p`: input
dimension (number of normalized states + structural parameters), hidden-layer
widths, number of policy outputs, and the initial bias of the sigmoid output
layer (e.g. the logit of a steady-state share, so the untrained policy starts
near the steady state instead of at 0.5).
"""
function policy_spec end

"""
    sample_batch(p, rng, batch_size) -> batch::NamedTuple

Draw one Float32 training batch over the model's normalized state × parameter
box. The returned named tuple must carry the `input_dim × batch_size`
network-input matrix in its `inputs` field; every other field is the model's
own business (whatever `training_loss` needs to evaluate the residuals).
Sampling happens on the CPU (reproducible for a given `rng`); the engine
moves the batch to the training device with `device(batch)`.
"""
function sample_batch end

"""
    training_loss(p, model, batch, quad; loss_kwargs...) -> Real

Scalar training objective — e.g. mean squared normalized equilibrium-condition
residuals plus penalties — differentiated through by [`train!`](@ref). This is
the function to edit (or re-dispatch via a new params type) to change what the
network learns; the engine never inspects its structure.
"""
function training_loss end

"""
    validation_report(p, model, batch, quad; loss_kwargs...) -> NamedTuple

Loss components on the fixed validation batch, logged during training. Must
contain `loss` — the early-stopping criterion, on the same scale as
[`training_loss`](@ref); any further fields are logged as-is. The fallback
reports the training loss only.
"""
validation_report(p, model, batch, quad; kwargs...) =
    (; loss=training_loss(p, model, batch, quad; kwargs...))

"""
    build_validation_panel(p, n_cases, seed)

Optional benchmark panel (e.g. calibrations with their grid-benchmark
policies) solved once before training and re-evaluated every `eval_every`
epochs — diagnostic only, never a training signal. The fallback returns an
empty panel, disabling the diagnostic.
"""
build_validation_panel(p, n_cases, seed) = ()

"""
    evaluate_validation_panel(p, solver, panel, T) -> NamedTuple or nothing

Compact solver-vs-benchmark metrics over `panel`, logged during training next
to the validation loss. Return `nothing` (the fallback) when there is no
panel.
"""
evaluate_validation_panel(p, solver, panel, T) = nothing
