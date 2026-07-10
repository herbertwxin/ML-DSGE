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
    NNSolver(p::RBCParams; lr=5e-4)

Policy network + optimizer state + quadrature for the Euler expectation.
The `RBCParams` carries both the reference calibration and the structural
bounds the network is trained over.
"""
struct NNSolver{M<:Chain,O}
    p::RBCParams
    model::M
    opt_state::O
    quad::Quadrature
end

function NNSolver(p::RBCParams; lr::Float64=5e-4)
    frac_ss = steady_state_share(p)
    model = build_policy_net(8, [64, 64, 64, 64], log(frac_ss / (1.0 - frac_ss)))
    return NNSolver(p, model, Flux.setup(Adam(lr), model), Quadrature(7))
end

"""
    NNPolicy(solver::NNSolver, p::RBCParams)

The network evaluated at one fixed structural calibration: pre-computes the
normalized parameter entries and the `(k, A)` normalization box so the policy
can be queried pointwise with `consumption_share(policy, k, A)` (and hence
run through the shared [`simulate`](@ref)).
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
    return NNPolicy(solver.model, theta, k_low, k_high, A_low, A_high)
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
    compute_residuals(model, batch, quad::Quadrature)

Normalized Euler-equation residual for a batch from [`sample_batch`](@ref):

    resid = (E[beta * u'(c') * R'] - u'(c)) / u'(c)

The raw residual is in marginal-utility units, whose magnitude varies by
orders of magnitude across the sampled `(gamma, c)` range; an unnormalized MSE
would be dominated by whichever draws have the most extreme curvature,
starving the hard (high beta / low delta / high gamma) region of gradient
signal. Dividing by `u'(c)` makes the residual a dimensionless relative Euler
error so all draws contribute on a comparable scale.
"""
function compute_residuals(model, batch, quad::Quadrature)
    (; inputs, k, A, k_low, k_high, A_low, A_high,
       alpha, beta, delta, rho, gamma, sigma_eps) = batch

    frac = vec(model(inputs))
    res = @. A * k^alpha + (1.0 - delta) * k
    c = max.(frac .* res, 1e-6)
    k1 = max.(res .- c, 1e-8)
    mu = c .^ (-gamma)

    k1_norm = normalize01(k1, k_low, k_high)
    logA = log.(A)
    theta_rows = inputs[3:8, :]

    rhs = zero(mu)
    for (z, w) in zip(quad.nodes, quad.weights)
        A1 = @. exp(rho * logA + sigma_eps * z)
        A1_norm = normalize01(A1, A_low, A_high)
        inputs1 = vcat(reshape(k1_norm, 1, :), reshape(A1_norm, 1, :), theta_rows)
        frac1 = vec(model(inputs1))
        res1 = @. A1 * k1^alpha + (1.0 - delta) * k1
        c1 = max.(frac1 .* res1, 1e-6)
        R1 = @. alpha * A1 * k1^(alpha - 1.0) + (1.0 - delta)
        rhs = rhs .+ w .* beta .* (c1 .^ (-gamma)) .* R1
    end

    return (rhs .- mu) ./ mu
end

euler_loss(model, batch, quad) = mean(abs2, compute_residuals(model, batch, quad))

"""
    train!(solver::NNSolver; batch_size=2048, epochs=50_000, eval_every=200,
           val_batch_size=8192, patience=20, min_rel_improve=5e-3,
           panel_n_cases=4, panel_T=120, panel_seed=321,
           best_checkpoint_path=nothing, rng=Random.default_rng())

Minimize the mean squared normalized Euler residual over freshly sampled
batches, with early stopping on a fixed validation batch. Every `eval_every`
epochs a fixed NN-vs-TI panel (TI policies solved once up front) is
re-simulated for rollout diagnostics. Returns the per-epoch training losses;
the solver's model holds the best-validation weights on exit.
"""
function train!(solver::NNSolver;
                batch_size::Int=2048, epochs::Int=50_000, eval_every::Int=200,
                val_batch_size::Int=8192, patience::Int=20, min_rel_improve::Float64=5e-3,
                panel_n_cases::Int=4, panel_T::Int=120, panel_seed::Int=321,
                best_checkpoint_path::Union{Nothing,String}=nothing,
                rng::AbstractRNG=Random.default_rng())
    losses = Float64[]
    val_batch = sample_batch(solver.p, rng, val_batch_size)
    panel = build_validation_panel(solver.p, panel_n_cases, panel_seed)
    isempty(panel) || @info "Validation panel ready" n_cases = length(panel) panel_T

    best_val_loss = Inf
    best_epoch = 0
    bad_evals = 0
    best_state = nothing

    @info "Training on the normalized Euler residual over the structural-parameter box..."
    for epoch in 1:epochs
        batch = sample_batch(solver.p, rng, batch_size)
        loss_val, grads = Flux.withgradient(m -> euler_loss(m, batch, solver.quad), solver.model)
        Flux.update!(solver.opt_state, solver.model, grads[1])
        push!(losses, loss_val)

        epoch % eval_every == 0 || continue

        val_loss = euler_loss(solver.model, val_batch, solver.quad)
        panel_metrics = evaluate_validation_panel(solver, panel, panel_T)
        if panel_metrics === nothing
            @info "epoch" epoch train_mse = loss_val val_mse = val_loss
        else
            lr = panel_metrics["level_ratio"]
            @info "epoch" epoch train_mse = loss_val val_mse = val_loss mean_nrmse = panel_metrics["aggregate"]["mean_nrmse"] max_nrmse = panel_metrics["aggregate"]["max_nrmse"] level_ratio_c = lr["consumption"] level_ratio_k = lr["capital"]
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

Save `Flux.state(model)` plus the `RBCParams` to a BSON file.
"""
function save_checkpoint(solver::NNSolver, path::AbstractString)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    model_state = Flux.state(solver.model)
    p = solver.p
    BSON.@save path model_state p
    return path
end

"""
    load_checkpoint(path) -> NNSolver

Restore an [`NNSolver`](@ref) saved by [`save_checkpoint`](@ref).
"""
function load_checkpoint(path::AbstractString)
    data = BSON.load(path, @__MODULE__)
    solver = NNSolver(data[:p])
    Flux.loadmodel!(solver.model, data[:model_state])
    @info "Loaded checkpoint" path
    return solver
end
