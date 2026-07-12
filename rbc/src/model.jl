# Model primitives: parameters, technology/preferences, steady state,
# productivity support, and Gauss-Hermite quadrature for the expectation
# in the Euler equation.

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

"""
    a_support_from_shock_params(rho, sigma_eps, a_sigma_mult, a_ss=1.0)

Productivity support `[A_low, A_high] = exp(±a_sigma_mult * sigma_stat) * a_ss`
with `sigma_stat = sigma_eps / sqrt(1 - rho^2)`. Shared by the NN state
normalization and the TI grid so both solvers see the same box.
"""
function a_support_from_shock_params(rho::Real, sigma_eps::Real, a_sigma_mult::Real, a_ss::Real=1.0)
    sigma_stat = sigma_eps / sqrt(max(1e-4, 1.0 - rho^2))
    w = a_sigma_mult * sigma_stat
    a_low = exp(-w) * a_ss
    return a_low, max(exp(w) * a_ss, a_low + 1e-6)
end

"Capital support `(k_low, k_high) = k_bounds .* k_ss(A=1)` for calibration `p`."
function k_support(p::RBCParams)
    k_ss, _, _ = steady_state_batch(p.alpha, p.beta, p.delta)
    return p.k_bounds[1] * k_ss, p.k_bounds[2] * k_ss
end

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

"""
    Quadrature(n)

Gauss-Hermite nodes/weights rescaled for expectations over a standard-normal
innovation: `E[f(eps)] ≈ sum(w .* f.(nodes))`.
"""
struct Quadrature
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function Quadrature(n::Int)
    x, w = gausshermite(n)
    return Quadrature(x .* sqrt(2), w ./ sqrt(pi))
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
