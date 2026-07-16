# Policy-network construction and the NNSolver container. Model-agnostic:
# the architecture comes from the model's `policy_spec`.

normalize01(x, lo, hi) = (x .- lo) ./ (hi .- lo)
denormalize01(x, lo, hi) = x .* (hi .- lo) .+ lo
normalize_scalar(x::Real, (lo, hi)::Tuple) = (x - lo) / (hi - lo)

"""
    a_support_from_shock_params(rho, sigma_eps, a_sigma_mult, a_ss=1.0)

Support `[A_low, A_high] = exp(±a_sigma_mult * sigma_stat) * a_ss` of an
AR(1)-in-logs exogenous state, with `sigma_stat = sigma_eps / sqrt(1 - rho^2)`.
Model-agnostic: shared by NN state normalization and benchmark grids so all
solvers of a model see the same box.
"""
function a_support_from_shock_params(rho::Real, sigma_eps::Real, a_sigma_mult::Real, a_ss::Real=1.0)
    sigma_stat = sigma_eps / sqrt(max(1e-4, 1.0 - rho^2))
    w = a_sigma_mult * sigma_stat
    a_low = exp(-w) * a_ss
    return a_low, max(exp(w) * a_ss, a_low + 1e-6)
end

"""
    build_policy_net(input_dim, hidden_dims, output_dim, output_bias)

`Dense(..., elu)` stack with a final `Dense(..., output_dim, sigmoid)` layer
whose bias is pre-set to `output_bias` (a scalar, or a vector with one entry
per output), so the untrained policy starts where the model wants it —
typically the steady state.
"""
function build_policy_net(input_dim::Int, hidden_dims::Vector{Int}, output_dim::Int, output_bias)
    dims = [input_dim; hidden_dims]
    hidden = (Dense(dims[i], dims[i+1], elu) for i in 1:length(hidden_dims))
    out = Dense(dims[end], output_dim, sigmoid)
    out.bias .= output_bias
    return Chain(hidden..., out)   # Flux default: Float32
end

"""
    NNSolver(p; lr=5e-4, n_quad=7, device=select_device(), model_state=nothing)

Policy network + optimizer state + quadrature for the expectation in the
model's equilibrium conditions. `p` is the model's parameter struct: it
carries the reference calibration and the training bounds, and its *type*
selects the model everywhere — architecture via [`policy_spec`](@ref),
batches via [`sample_batch`](@ref), objective via [`training_loss`](@ref).

The model lives on `device` (see [`select_device`](@ref)) in Float32.
`model_state` — a CPU `Flux.state` from a checkpoint — is loaded before the
device move, so checkpoints stay device-independent (older Float64
checkpoints are converted on load by `Flux.loadmodel!`).
"""
struct NNSolver{P,M<:Chain,O,D}
    p::P
    model::M
    opt_state::O
    quad::Quadrature
    device::D
end

function NNSolver(p; lr::Float64=5e-4, n_quad::Int=7, device=select_device(), model_state=nothing)
    spec = policy_spec(p)
    model = build_policy_net(spec.input_dim, spec.hidden, spec.output_dim, spec.output_bias)
    model_state === nothing || Flux.loadmodel!(model, model_state)
    model = model |> device
    return NNSolver(p, model, Flux.setup(Adam(lr), model), Quadrature(n_quad), device)
end
