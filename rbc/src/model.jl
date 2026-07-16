# The RBC model: everything the engine does not know about.
#
# First the economics — parameters, technology/preferences, steady state,
# productivity support — then the model's implementation of the engine
# interface (see engine/interface.jl): network spec, batch sampling, the
# Euler-residual loss, the fixed-calibration policy wrapper, and the NN-vs-TI
# validation panel. Swapping the training loss means editing `euler_terms` /
# `training_loss` here; no engine file is involved.

"""
    RBCParams

Structural parameters plus training/sampling bounds for the stochastic RBC
model. The `*_bounds` fields define the box over which the neural solver is
trained; the scalar fields are the calibration used when a single economy is
solved or simulated.

Field order is part of the checkpoint format (`rbc_nn.bson` stores this struct
via BSON) — append new fields at the end if the struct ever grows.
"""
Base.@kwdef struct RBCParams
    alpha::Float64 = 0.30      # capital share
    beta::Float64  = 0.95      # discount factor
    delta::Float64 = 0.1       # depreciation rate
    gamma::Float64 = 2.0       # risk aversion (CRRA)
    rho::Float64   = 0.90      # persistence of productivity shock
    sigma_eps::Float64 = 0.02  # std dev of shock innovation

    # State-space box: k as a fraction of the (A = 1) steady-state capital.
    k_bounds::Tuple{Float64,Float64} = (0.5, 1.5)

    # Productivity box half-width in stationary std devs:
    # log A in ±A_sigma_mult * sigma_stat, sigma_stat = sigma_eps / sqrt(1 - rho^2).
    A_sigma_mult::Float64 = 3.0

    # Structural-parameter box the NN learns over.
    alpha_bounds::Tuple{Float64,Float64} = (0.20, 0.45)
    beta_bounds::Tuple{Float64,Float64}  = (0.90, 0.99)
    delta_bounds::Tuple{Float64,Float64} = (0.02, 0.15)
    rho_bounds::Tuple{Float64,Float64}   = (0.85, 0.99)
    gamma_bounds::Tuple{Float64,Float64} = (0.5, 4.0)
    sigma_eps_bounds::Tuple{Float64,Float64} = (0.005, 0.05)

    # Oversampling of hard parameter corners in training batches: high beta,
    # low delta, and gamma near either edge of its normalized range.
    hard_region_prob::Float64 = 0.5
    hard_beta_low_norm::Float64 = 0.85
    hard_delta_high_norm::Float64 = 0.20
end

# ---------------------------------------------------------------------------
# Technology and preferences. Broadcastable over any argument.
# ---------------------------------------------------------------------------

"Output `y = A k^alpha`."
production(k, A, alpha) = A * k^alpha

"Total resources available for consumption + next-period capital."
resources(k, A, alpha, delta) = production(k, A, alpha) + (1 - delta) * k

"Gross return on capital `R = alpha A k^(alpha-1) + 1 - delta`."
gross_return(k, A, alpha, delta) = alpha * A * k^(alpha - 1) + (1 - delta)

"CRRA marginal utility `u'(c) = c^(-gamma)`."
marginal_utility(c, gamma) = c^(-gamma)

"Next-period log productivity: `log A' = rho log A + sigma_eps * eps`."
next_productivity(A, eps, rho, sigma_eps) = exp(rho * log(A) + sigma_eps * eps)

# ---------------------------------------------------------------------------
# Steady state and state-space support
# ---------------------------------------------------------------------------

"""
    steady_state_batch(alpha, beta, delta) -> (k_ss, c_ss, y_ss)

Elementwise (A = 1) deterministic steady state; works on scalars and arrays
alike via broadcasting.
"""
function steady_state_batch(alpha, beta, delta)
    # Integer literals keep the computation in the input eltype (Float32 for
    # NN training batches, Float64 for the TI benchmark) without promotion.
    term = @. (1 / beta - (1 - delta)) / alpha
    k_ss = @. term^(1 / (alpha - 1))
    y_ss = @. k_ss^alpha
    c_ss = @. y_ss - delta * k_ss
    return k_ss, c_ss, y_ss
end

"""
    steady_state(p::RBCParams) -> (k, c, y, A)

Deterministic steady state at `A = 1`, as a named tuple.
"""
function steady_state(p::RBCParams)
    k, c, y = steady_state_batch(p.alpha, p.beta, p.delta)
    return (k=k, c=c, y=y, A=1.0)
end

"Steady-state consumption share of total resources (used to seed both solvers)."
function steady_state_share(p::RBCParams)
    ss = steady_state(p)
    return ss.c / resources(ss.k, ss.A, p.alpha, p.delta)
end

"Capital support `(k_low, k_high) = k_bounds .* k_ss(A=1)` for calibration `p`."
function k_support(p::RBCParams)
    k_ss, _, _ = steady_state_batch(p.alpha, p.beta, p.delta)
    return p.k_bounds[1] * k_ss, p.k_bounds[2] * k_ss
end

# ---------------------------------------------------------------------------
# Parameter sampling and (de)serialization
# ---------------------------------------------------------------------------

"""
    sample_params_uniform(base::RBCParams, rng) -> RBCParams

Draw one structural parameter set uniformly from `base`'s training bounds,
keeping all bounds/config fields fixed.
"""
function sample_params_uniform(base::RBCParams, rng::AbstractRNG)
    unif((lo, hi)) = lo + rand(rng) * (hi - lo)
    structural = (
        alpha=unif(base.alpha_bounds), beta=unif(base.beta_bounds), delta=unif(base.delta_bounds),
        gamma=unif(base.gamma_bounds), rho=unif(base.rho_bounds), sigma_eps=unif(base.sigma_eps_bounds),
    )
    return with_calibration(base, structural)
end

"""
    with_calibration(base::RBCParams, structural::NamedTuple) -> RBCParams

Copy of `base` with the six structural scalars replaced (bounds/config kept).
"""
function with_calibration(base::RBCParams, structural::NamedTuple)
    fields = (f => get(structural, f, getfield(base, f)) for f in fieldnames(RBCParams))
    return RBCParams(; fields...)
end

"""
    params_to_dict(p::RBCParams) -> Dict{String,Any}

JSON-friendly representation of `p` (tuples become vectors).
"""
function params_to_dict(p::RBCParams)
    return Dict{String,Any}(
        string(f) => (v = getfield(p, f); v isa Tuple ? collect(v) : v)
        for f in fieldnames(RBCParams)
    )
end

"""
    params_from_dict(d) -> RBCParams

Inverse of [`params_to_dict`](@ref); accepts any `AbstractDict` with string keys.
"""
function params_from_dict(d)
    kwargs = Dict{Symbol,Any}()
    for f in fieldnames(RBCParams)
        haskey(d, string(f)) || continue
        v = d[string(f)]
        kwargs[f] = v isa AbstractVector ? Tuple(Float64.(v)) : v
    end
    return RBCParams(; kwargs...)
end

# ===========================================================================
# Engine interface: network spec, batch sampling, training loss
# ===========================================================================

"""
    policy_spec(p::RBCParams)

Architecture of the RBC policy network: a normalized 8-vector
`(k, A, alpha, beta, delta, rho, gamma, sigma_eps)` in, a consumption share
in `(0, 1)` out (sigmoid), with the output bias pre-set so the untrained
policy starts at the steady-state share.
"""
function policy_spec(p::RBCParams)
    frac_ss = steady_state_share(p)
    return (input_dim=8, hidden=[64, 64, 64, 64], output_dim=1,
            output_bias=log(frac_ss / (1.0 - frac_ss)))
end

"""
    sample_batch(p::RBCParams, rng, batch_size)

Draw a Float32 training batch over the normalized state × parameter box. Most
draws are uniform; with probability `p.hard_region_prob`, a column instead
uses high beta, low delta, and gamma sampled from either edge of its range.
Returns the `8 × batch_size` network input matrix plus the physical states,
structural parameters, and per-sample `(k, A)` support needed to evaluate the
Euler residual.
"""
function sample_batch(p::RBCParams, rng::AbstractRNG, batch_size::Int)
    # Rows: k, A, alpha, beta, delta, rho, gamma, sigma_eps (all in [0, 1]).
    inputs = rand(rng, Float32, 8, batch_size)

    if p.hard_region_prob > 0.0
        beta_low = Float32(p.hard_beta_low_norm)
        delta_high = Float32(p.hard_delta_high_norm)
        gamma_width = 0.20f0  # outer 20% of normalized gamma, split low/high
        for col in axes(inputs, 2)
            if rand(rng) < p.hard_region_prob
                inputs[4, col] = beta_low + (1f0 - beta_low) * rand(rng, Float32)
                inputs[5, col] = delta_high * rand(rng, Float32)
                inputs[7, col] = rand(rng, Bool) ?
                    gamma_width * rand(rng, Float32) :
                    1f0 - gamma_width * rand(rng, Float32)
            end
        end
    end

    b32(bounds) = Float32.(bounds)   # RBCParams stores Float64 bounds
    alpha     = denormalize01(inputs[3, :], b32(p.alpha_bounds)...)
    beta      = denormalize01(inputs[4, :], b32(p.beta_bounds)...)
    delta     = denormalize01(inputs[5, :], b32(p.delta_bounds)...)
    rho       = denormalize01(inputs[6, :], b32(p.rho_bounds)...)
    gamma     = denormalize01(inputs[7, :], b32(p.gamma_bounds)...)
    sigma_eps = denormalize01(inputs[8, :], b32(p.sigma_eps_bounds)...)

    A_mult = Float32(p.A_sigma_mult)
    sigma_stat = @. sigma_eps / sqrt(max(1f-4, 1 - rho^2))
    A_low = @. exp(-A_mult * sigma_stat)
    A_high = @. max(exp(A_mult * sigma_stat), A_low + 1f-6)

    k_ss, _, _ = steady_state_batch(alpha, beta, delta)
    k_low = Float32(p.k_bounds[1]) .* k_ss
    k_high = Float32(p.k_bounds[2]) .* k_ss

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

    frac = vec(model(inputs))
    res = @. A * k^alpha + (1 - delta) * k
    c = max.(frac .* res, 1f-6)
    k1 = max.(res .- c, 1f-8)
    mu = c .^ (-gamma)

    k1_norm = normalize01(k1, k_low, k_high)
    logA = log.(A)
    theta_rows = inputs[3:8, :]

    # Device-resident quadrature rows (1×Q), kept off the AD tape. Built with
    # `similar(k, ...)` so they live wherever the batch lives.
    Q = length(quad.nodes)
    zrow, wrow, onesrow = Flux.Zygote.ignore() do
        z = copyto!(similar(k, 1, Q), reshape(Float32.(quad.nodes), 1, Q))
        w = copyto!(similar(k, 1, Q), reshape(Float32.(quad.weights), 1, Q))
        o = fill!(similar(k, 1, Q), 1f0)
        (z, w, o)
    end

    # All Q quadrature nodes go through the network in ONE call: an 8×(B*Q)
    # input whose q-th block of B columns is node q. This is a small model, so
    # one fat matmul keeps a GPU busy where Q separate skinny calls (and their
    # backward passes) leave it idle between kernel launches. B×Q matrices
    # broadcast the per-sample vectors down columns and the per-node rows
    # across; `reshape(·, 1, :)` flattens them in exactly the block order the
    # network input needs (sample index fastest, node index by block).
    A1 = @. exp(rho * logA + sigma_eps * zrow)              # B×Q
    A1_norm = @. (A1 - A_low) / (A_high - A_low)            # B×Q
    k1_rep = k1_norm .* onesrow                             # B×Q, equal columns
    inputs1 = vcat(reshape(k1_rep, 1, :), reshape(A1_norm, 1, :),
                   repeat(theta_rows, 1, Q))                # 8×(B*Q)
    frac1 = reshape(model(inputs1), :, Q)                   # B×Q
    res1 = @. A1 * k1^alpha + (1 - delta) * k1
    c1 = max.(frac1 .* res1, 1f-6)
    R1 = @. alpha * A1 * k1^(alpha - 1) + (1 - delta)
    rhs = beta .* vec(sum(@.(c1^(-gamma) * R1 * wrow); dims=2))

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
    # Float32 weight keeps the loss scalar (and hence the backward seed) Float32.
    return mean(abs2, resid) + Float32(k_oob_weight) * mean(k_oob)
end

"The RBC training objective handed to the engine; see [`euler_loss`](@ref)."
training_loss(p::RBCParams, model, batch, quad::Quadrature; k_oob_weight::Real=1.0) =
    euler_loss(model, batch, quad; k_oob_weight)

"Validation-batch loss components: penalized loss, Euler MSE, over-saving penalty."
function validation_report(p::RBCParams, model, batch, quad::Quadrature; k_oob_weight::Real=1.0)
    resid, k_oob = euler_terms(model, batch, quad)
    euler_mse = mean(abs2, resid)
    oob = mean(k_oob)
    return (loss=euler_mse + k_oob_weight * oob, euler_mse=euler_mse, k_oob=oob)
end

# ---------------------------------------------------------------------------
# Fixed-calibration policy wrapper (for the shared simulate)
# ---------------------------------------------------------------------------

"""
    NNPolicy(solver::NNSolver, p::RBCParams)

The network evaluated at one fixed structural calibration: pre-computes the
normalized parameter entries and the `(k, A)` normalization box so the policy
can be queried pointwise with `consumption_share(policy, k, A)` (and hence
run through the shared [`simulate`](@ref)). Simulation is a scalar
state-by-state loop in Float64, so the policy uses a CPU copy of the model
(pointwise GPU calls would be far slower than the copy) and casts the state
to Float32 only at the network input.
"""
struct NNPolicy{M}
    model::M
    theta_norm::NTuple{6,Float64}
    k_low::Float64
    k_high::Float64
    A_low::Float64
    A_high::Float64
end

function NNPolicy(solver::NNSolver{RBCParams}, p::RBCParams)
    base = solver.p
    k_low, k_high = k_support(p)
    A_low, A_high = a_support_from_shock_params(p.rho, p.sigma_eps, base.A_sigma_mult)
    theta = (normalize_scalar(p.alpha, base.alpha_bounds),
             normalize_scalar(p.beta, base.beta_bounds),
             normalize_scalar(p.delta, base.delta_bounds),
             normalize_scalar(p.rho, base.rho_bounds),
             normalize_scalar(p.gamma, base.gamma_bounds),
             normalize_scalar(p.sigma_eps, base.sigma_eps_bounds))
    return NNPolicy(cpu(solver.model), theta, k_low, k_high, A_low, A_high)
end

function consumption_share(pol::NNPolicy, k::Real, A::Real)
    kn = (k - pol.k_low) / (pol.k_high - pol.k_low)
    an = (A - pol.A_low) / (pol.A_high - pol.A_low)
    x = Float32[kn, an, pol.theta_norm...]
    return pol.model(reshape(x, 8, 1))[1]
end

"Simulate the trained network at calibration `p` (shared loop, same shocks API)."
simulate(solver::NNSolver{RBCParams}, p::RBCParams; kwargs...) =
    simulate(NNPolicy(solver, p), p; kwargs...)

# ---------------------------------------------------------------------------
# Validation panel (NN vs TI, diagnostic only)
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
    evaluate_validation_panel(p::RBCParams, solver, panel, T)

Average NN-vs-TI [`gap_metrics`](@ref) across the panel (identical shock
seeds), condensed to the fields logged during training.
"""
function evaluate_validation_panel(p::RBCParams, solver::NNSolver, panel, T::Int)
    isempty(panel) && return nothing

    metrics_list = map(panel) do item
        nn_res = simulate(solver, item.params; T, rng=Xoshiro(item.seed))
        ti_res = simulate(item.policy; T, rng=Xoshiro(item.seed))
        gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))
    end

    return (
        mean_nrmse=mean(m["aggregate"]["mean_nrmse"] for m in metrics_list),
        max_nrmse=maximum(m["aggregate"]["max_nrmse"] for m in metrics_list),
        level_ratio_c=mean(m["consumption"]["level_ratio"] for m in metrics_list),
        level_ratio_k=mean(m["capital"]["level_ratio"] for m in metrics_list),
    )
end
