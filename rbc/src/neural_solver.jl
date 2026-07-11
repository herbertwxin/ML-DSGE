# Neural-network Euler-residual solver.
#
# The policy network maps a normalized 8-vector
# (k, A, alpha, beta, delta, rho, gamma, sigma_eps) to a consumption share in
# (0, 1) via a sigmoid output, and is trained on the *normalized* (relative)
# Euler-equation residual over the whole structural-parameter box — see
# `compute_residuals` for why the residual is divided by marginal utility.

normalize01(x, lo, hi) = (x .- lo) ./ (hi .- lo)
denormalize01(x, lo, hi) = x .* (hi .- lo) .+ lo
normalize_scalar(x::Real, (lo, hi)::Tuple) = (x - lo) / (hi - lo)

# ---------------------------------------------------------------------------
# Device selection (CPU / NVIDIA CUDA / Apple Metal)
# ---------------------------------------------------------------------------

"""
    select_device(preference=:auto)

Pick the compute device for NN training. `:auto` returns the first functional
GPU backend whose trigger package is loaded (CUDA on NVIDIA, Metal on Apple —
see the conditional loading in `FullRBC.jl`), falling back to the CPU;
`:cpu` forces the CPU; `:gpu` errors if no functional GPU is found.
"""
function select_device(preference::Symbol=:auto)
    preference === :cpu && return cpu_device()
    dev = gpu_device()
    if preference === :gpu && dev isa typeof(cpu_device())
        error("device=:gpu requested but no functional GPU backend is available. " *
              "Install/load CUDA.jl (NVIDIA) or Metal.jl (Apple) and check `gpu_device()`.")
    end
    return dev
end

# Apple GPUs have no Float64 support (and Float32 is much faster on NVIDIA
# too), so training runs in Float32 on any GPU and Float64 on the CPU.
train_eltype(device) = device isa typeof(cpu_device()) ? Float64 : Float32

"""
    to_device(batch::NamedTuple, device, T) -> NamedTuple

Convert every array in a [`sample_batch`](@ref) result to element type `T`
and move it to `device`. Batches are always *sampled* on the CPU in Float64
(keeping draws reproducible for a given `rng` regardless of device), then
transferred.
"""
to_device(batch::NamedTuple, device, ::Type{T}) where {T} =
    map(x -> x isa AbstractArray ? device(convert(AbstractArray{T}, x)) : x, batch)

"""
    build_policy_net(input_dim, hidden_dims, output_bias)

`Dense(..., elu)` stack with a final `Dense(..., 1, sigmoid)` layer whose bias
is pre-set to `output_bias`, so the untrained policy starts near the
steady-state consumption share instead of at 0.5.
"""
function build_policy_net(input_dim::Int, hidden_dims::Vector{Int}, output_bias::Float64)
    dims = [input_dim; hidden_dims]
    hidden = (Dense(dims[i], dims[i+1], elu) for i in 1:length(hidden_dims))
    out = Dense(dims[end], 1, sigmoid)
    out.bias .= output_bias
    return Chain(hidden..., out) |> f64
end

"""
    NNSolver(p::RBCParams; lr=5e-4, device=select_device(), model_state=nothing)

Policy network + optimizer state + quadrature for the Euler expectation.
The `RBCParams` carries both the reference calibration and the structural
bounds the network is trained over.

The model lives on `device` (see [`select_device`](@ref)) in the matching
precision ([`train_eltype`](@ref): Float32 on GPU, Float64 on CPU).
`model_state` — a CPU/Float64 `Flux.state` from a checkpoint — is loaded
before the device/precision move, so checkpoints stay device-independent.
"""
struct NNSolver{M<:Chain,O,D}
    p::RBCParams
    model::M
    opt_state::O
    quad::Quadrature
    device::D
end

function NNSolver(p::RBCParams; lr::Float64=5e-4, device=select_device(), model_state=nothing)
    frac_ss = steady_state_share(p)
    model = build_policy_net(8, [64, 64, 64, 64], log(frac_ss / (1.0 - frac_ss)))
    model_state === nothing || Flux.loadmodel!(model, model_state)
    T = train_eltype(device)
    model = (T === Float32 ? f32(model) : model) |> device
    return NNSolver(p, model, Flux.setup(Adam(lr), model), Quadrature(7), device)
end

"""
    NNPolicy(solver::NNSolver, p::RBCParams)

The network evaluated at one fixed structural calibration: pre-computes the
normalized parameter entries and the `(k, A)` normalization box so the policy
can be queried pointwise with `consumption_share(policy, k, A)` (and hence
run through the shared [`simulate`](@ref)). Simulation is a scalar
state-by-state loop, so the policy always uses a CPU/Float64 copy of the
model — pointwise GPU calls would be far slower than the copy.
"""
struct NNPolicy{M}
    model::M
    theta_norm::NTuple{6,Float64}
    k_low::Float64
    k_high::Float64
    A_low::Float64
    A_high::Float64
end

function NNPolicy(solver::NNSolver, p::RBCParams)
    base = solver.p
    k_low, k_high = k_support(p)
    A_low, A_high = a_support_from_shock_params(p.rho, p.sigma_eps, base.A_sigma_mult)
    theta = (normalize_scalar(p.alpha, base.alpha_bounds),
             normalize_scalar(p.beta, base.beta_bounds),
             normalize_scalar(p.delta, base.delta_bounds),
             normalize_scalar(p.rho, base.rho_bounds),
             normalize_scalar(p.gamma, base.gamma_bounds),
             normalize_scalar(p.sigma_eps, base.sigma_eps_bounds))
    return NNPolicy(f64(cpu(solver.model)), theta, k_low, k_high, A_low, A_high)
end

function consumption_share(pol::NNPolicy, k::Real, A::Real)
    kn = (k - pol.k_low) / (pol.k_high - pol.k_low)
    an = (A - pol.A_low) / (pol.A_high - pol.A_low)
    x = [kn, an, pol.theta_norm...]
    return pol.model(reshape(x, 8, 1))[1]
end

"Simulate the trained network at calibration `p` (shared loop, same shocks API)."
simulate(solver::NNSolver, p::RBCParams; kwargs...) = simulate(NNPolicy(solver, p), p; kwargs...)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

"""
    sample_batch(p::RBCParams, rng, batch_size)

Draw a training batch uniformly over the normalized state × parameter box.
Returns the `8 × batch_size` network input matrix plus the physical states,
structural parameters, and per-sample `(k, A)` support needed to evaluate the
Euler residual.
"""
function sample_batch(p::RBCParams, rng::AbstractRNG, batch_size::Int)
    # Rows: k, A, alpha, beta, delta, rho, gamma, sigma_eps (all in [0, 1]).
    inputs = rand(rng, 8, batch_size)

    if p.hard_region_prob > 0.0
        for col in axes(inputs, 2)
            if rand(rng) < p.hard_region_prob
                inputs[4, col] = p.hard_beta_low_norm + (1.0 - p.hard_beta_low_norm) * rand(rng)
                inputs[5, col] = p.hard_delta_high_norm * rand(rng)
            end
        end
    end

    alpha     = denormalize01(inputs[3, :], p.alpha_bounds...)
    beta      = denormalize01(inputs[4, :], p.beta_bounds...)
    delta     = denormalize01(inputs[5, :], p.delta_bounds...)
    rho       = denormalize01(inputs[6, :], p.rho_bounds...)
    gamma     = denormalize01(inputs[7, :], p.gamma_bounds...)
    sigma_eps = denormalize01(inputs[8, :], p.sigma_eps_bounds...)

    sigma_stat = @. sigma_eps / sqrt(max(1e-4, 1.0 - rho^2))
    A_low = @. exp(-p.A_sigma_mult * sigma_stat)
    A_high = @. max(exp(p.A_sigma_mult * sigma_stat), A_low + 1e-6)

    k_ss, _, _ = steady_state_batch(alpha, beta, delta)
    k_low = p.k_bounds[1] .* k_ss
    k_high = p.k_bounds[2] .* k_ss

    k = denormalize01(inputs[1, :], k_low, k_high)
    A = denormalize01(inputs[2, :], A_low, A_high)

    return (inputs=inputs, k=k, A=A, k_low=k_low, k_high=k_high, A_low=A_low, A_high=A_high,
            alpha=alpha, beta=beta, delta=delta, rho=rho, gamma=gamma, sigma_eps=sigma_eps)
end

"""
    euler_terms(model, batch, quad::Quadrature) -> (resid, k_oob)

Per-sample ingredients of the training loss for a batch from
[`sample_batch`](@ref):

- `resid`: normalized Euler residual
  `(E[beta * u'(c') * R'] - u'(c)) / u'(c)`. The raw residual is in
  marginal-utility units, whose magnitude varies by orders of magnitude
  across the sampled `(gamma, c)` range; an unnormalized MSE would be
  dominated by whichever draws have the most extreme curvature. Dividing by
  `u'(c)` makes it a dimensionless relative Euler error so all draws
  contribute on a comparable scale.

- `k_oob`: squared overshoot of *normalized* next-period capital outside
  `[0, 1]` — the over-saving (transversality) penalty. The Euler equation is
  only a first-order condition: a continuum of over-saving policies (consume
  a vanishing share, accumulate capital without bound, consumption growth
  tracking `(beta*R)^(1/gamma)`) satisfies it pointwise while violating
  transversality, and the residual — computed with the network supplying its
  own continuation — cannot tell them from the true policy. Those spurious
  solutions must drive `k'` out of the state box, so penalizing out-of-box
  `k'` removes them from the feasible set without referencing any external
  benchmark; inside the box the penalty is identically zero.
"""
function euler_terms(model, batch, quad::Quadrature)
    (; inputs, k, A, k_low, k_high, A_low, A_high,
       alpha, beta, delta, rho, gamma, sigma_eps) = batch

    # Literals/constants must be in the batch's element type: a Float64
    # scalar broadcast over Float32 GPU arrays would promote everything to
    # Float64, which Apple GPUs cannot execute at all.
    T = eltype(k)
    c_floor = T(1e-6)
    k_floor = T(1e-8)

    frac = vec(model(inputs))
    res = @. A * k^alpha + (1 - delta) * k
    c = max.(frac .* res, c_floor)
    k1 = max.(res .- c, k_floor)
    mu = c .^ (-gamma)

    k1_norm = normalize01(k1, k_low, k_high)
    logA = log.(A)
    theta_rows = inputs[3:8, :]

    rhs = zero(mu)
    for (zn, wn) in zip(quad.nodes, quad.weights)
        z, w = T(zn), T(wn)
        A1 = @. exp(rho * logA + sigma_eps * z)
        A1_norm = normalize01(A1, A_low, A_high)
        inputs1 = vcat(reshape(k1_norm, 1, :), reshape(A1_norm, 1, :), theta_rows)
        frac1 = vec(model(inputs1))
        res1 = @. A1 * k1^alpha + (1 - delta) * k1
        c1 = max.(frac1 .* res1, c_floor)
        R1 = @. alpha * A1 * k1^(alpha - 1) + (1 - delta)
        rhs = rhs .+ w .* beta .* (c1 .^ (-gamma)) .* R1
    end

    k_oob = @. max(k1_norm - 1, 0)^2 + max(-k1_norm, 0)^2
    return (rhs .- mu) ./ mu, k_oob
end

"Normalized Euler residuals only (diagnostics); see [`euler_terms`](@ref)."
compute_residuals(model, batch, quad::Quadrature) = euler_terms(model, batch, quad)[1]

"""
    euler_loss(model, batch, quad; k_oob_weight=1.0)

Training objective: mean squared normalized Euler residual plus
`k_oob_weight` times the mean over-saving penalty (see [`euler_terms`](@ref)).
"""
function euler_loss(model, batch, quad::Quadrature; k_oob_weight::Real=1.0)
    resid, k_oob = euler_terms(model, batch, quad)
    # Keep the loss scalar in the batch precision: a Float64 weight would
    # promote it and seed the backward pass with Float64 on the GPU.
    return mean(abs2, resid) + eltype(resid)(k_oob_weight) * mean(k_oob)
end

"""
    train!(solver::NNSolver; batch_size=2048, epochs=50_000, eval_every=200,
           val_batch_size=8192, patience=20, min_rel_improve=5e-3,
           k_oob_weight=1.0, panel_n_cases=4, panel_T=120, panel_seed=321,
           best_checkpoint_path=nothing, rng=Random.default_rng())

Minimize [`euler_loss`](@ref) — mean squared normalized Euler residual plus
the `k_oob_weight`-scaled over-saving penalty — over freshly sampled batches,
with early stopping on the same penalized loss evaluated on a fixed
validation batch. Every `eval_every` epochs a fixed NN-vs-TI panel (TI
policies solved once up front) is re-simulated and logged; the panel is
*diagnostic only* and plays no role in stopping or model selection, so the
network learns the model without ever referencing the TI benchmark. Returns
the per-epoch training losses; the solver's model holds the best-validation
weights on exit.
"""
function train!(solver::NNSolver;
                batch_size::Int=2048, epochs::Int=50_000, eval_every::Int=200,
                val_batch_size::Int=8192, patience::Int=20, min_rel_improve::Float64=5e-3,
                k_oob_weight::Float64=1.0,
                panel_n_cases::Int=4, panel_T::Int=120, panel_seed::Int=321,
                best_checkpoint_path::Union{Nothing,String}=nothing,
                rng::AbstractRNG=Random.default_rng())
    losses = Float64[]
    T = train_eltype(solver.device)
    @info "Training device" solver.device precision = T
    val_batch = to_device(sample_batch(solver.p, rng, val_batch_size), solver.device, T)
    panel = build_validation_panel(solver.p, panel_n_cases, panel_seed)
    isempty(panel) || @info "Validation panel ready" n_cases = length(panel) panel_T

    best_val_loss = Inf
    best_epoch = 0
    bad_evals = 0
    best_state = nothing

    @info "Training on normalized Euler residual + over-saving penalty..." k_oob_weight
    for epoch in 1:epochs
        batch = to_device(sample_batch(solver.p, rng, batch_size), solver.device, T)
        loss_val, grads = Flux.withgradient(
            m -> euler_loss(m, batch, solver.quad; k_oob_weight), solver.model)
        Flux.update!(solver.opt_state, solver.model, grads[1])
        push!(losses, loss_val)

        epoch % eval_every == 0 || continue

        val_resid, val_k_oob = euler_terms(solver.model, val_batch, solver.quad)
        val_mse = mean(abs2, val_resid)
        val_oob = mean(val_k_oob)
        val_loss = val_mse + k_oob_weight * val_oob
        panel_metrics = evaluate_validation_panel(solver, panel, panel_T)
        if panel_metrics === nothing
            @info "epoch" epoch train_loss = loss_val val_mse val_oob
        else
            lr = panel_metrics["level_ratio"]
            @info "epoch" epoch train_loss = loss_val val_mse val_oob mean_nrmse = panel_metrics["aggregate"]["mean_nrmse"] max_nrmse = panel_metrics["aggregate"]["max_nrmse"] level_ratio_c = lr["consumption"] level_ratio_k = lr["capital"]
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
# Validation panel
# ---------------------------------------------------------------------------

"""
    build_validation_panel(p::RBCParams, n_cases, seed)

Fixed panel of calibrations with their TI benchmark policies, solved once (in
parallel over Julia threads) and reused for cheap NN-vs-TI diagnostics during
training.
"""
function build_validation_panel(p::RBCParams, n_cases::Int, seed::Int)
    n_cases <= 0 && return NamedTuple{(:params, :policy, :seed)}[]
    rng = Xoshiro(seed)
    cases = [sample_params_uniform(p, rng) for _ in 1:n_cases]
    panel = Vector{Any}(undef, n_cases)
    Threads.@threads for i in 1:n_cases
        panel[i] = (params=cases[i], policy=solve(TISolver(cases[i])), seed=seed + i)
    end
    return panel
end

"""
    evaluate_validation_panel(solver::NNSolver, panel, T)

Average NN-vs-TI [`gap_metrics`](@ref) across the panel, simulating both
policies with identical shock seeds.
"""
function evaluate_validation_panel(solver::NNSolver, panel, T::Int)
    isempty(panel) && return nothing

    metrics_list = map(panel) do item
        nn_res = simulate(solver, item.params; T, rng=Xoshiro(item.seed))
        ti_res = simulate(item.policy; T, rng=Xoshiro(item.seed))
        gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))
    end

    series = ("consumption", "capital", "output", "investment")
    avg = Dict{String,Any}(
        s => Dict(
            "rmse" => mean(m[s]["rmse"] for m in metrics_list),
            "nrmse" => mean(m[s]["nrmse_vs_ti_std"] for m in metrics_list),
            "level_ratio" => mean(m[s]["level_ratio"] for m in metrics_list),
        ) for s in series
    )
    avg["level_ratio"] = Dict(s => avg[s]["level_ratio"] for s in series)
    avg["aggregate"] = Dict(
        "mean_nrmse" => mean(m["aggregate"]["mean_nrmse"] for m in metrics_list),
        "max_nrmse" => maximum(m["aggregate"]["max_nrmse"] for m in metrics_list),
    )
    return avg
end

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

"""
    save_checkpoint(solver::NNSolver, path)

Save `Flux.state(model)` plus the `RBCParams` to a BSON file. The state is
always written as CPU/Float64 arrays, so checkpoints are interchangeable
across devices (and identical in format to pre-GPU checkpoints).
"""
function save_checkpoint(solver::NNSolver, path::AbstractString)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    model_state = Flux.state(f64(cpu(solver.model)))
    p = solver.p
    BSON.@save path model_state p
    return path
end

"""
    load_checkpoint(path; device=cpu_device()) -> NNSolver

Restore an [`NNSolver`](@ref) saved by [`save_checkpoint`](@ref). Defaults to
the CPU — the checkpoint's consumers (`simulate`, `compare.jl`,
`diagnose_divergence.jl`) run pointwise where a GPU would only add transfer
overhead; pass `device=select_device()` to continue *training* on a GPU.
"""
function load_checkpoint(path::AbstractString; device=cpu_device())
    data = BSON.load(path, @__MODULE__)
    solver = NNSolver(data[:p]; device, model_state=data[:model_state])
    @info "Loaded checkpoint" path solver.device
    return solver
end
