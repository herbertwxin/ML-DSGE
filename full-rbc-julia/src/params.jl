# Model parameters and small shared helpers.
# Julia port of `learn_rbc.py::Params` / `a_support_from_shock_params` /
# `RBCTISolver.calculate_steady_state` / `RBCSolver._steady_state_batch`.

"""
    RBCParams

Structural parameters and training/sampling bounds for the RBC model.
Mirrors the Python `@dataclass Params` in `learn_rbc.py`.
"""
Base.@kwdef struct RBCParams
    alpha::Float64 = 0.30      # capital share
    beta::Float64  = 0.95      # discount factor
    delta::Float64 = 0.1       # depreciation rate
    gamma::Float64 = 2.0       # risk aversion (CRRA)
    rho::Float64   = 0.90      # persistence of productivity shock
    sigma_eps::Float64 = 0.02  # std dev of shock innovation

    # Bounds for state space: k as a fraction of steady-state capital.
    k_bounds::Tuple{Float64,Float64} = (0.5, 1.5)

    # How wide the NN's A normalization box is: log A in [-n, +n] * sigma_stat,
    # sigma_stat = sigma_eps / sqrt(1 - rho^2).
    A_sigma_mult::Float64 = 3.0

    # Bounds for STRUCTURAL PARAMETERS (NN learns over this whole space).
    alpha_bounds::Tuple{Float64,Float64} = (0.20, 0.45)
    beta_bounds::Tuple{Float64,Float64}  = (0.90, 0.99)
    delta_bounds::Tuple{Float64,Float64} = (0.02, 0.15)
    rho_bounds::Tuple{Float64,Float64}   = (0.85, 0.99)
    gamma_bounds::Tuple{Float64,Float64} = (0.5, 4.0)
    sigma_eps_bounds::Tuple{Float64,Float64} = (0.005, 0.05)

    # Oversampling settings for the "hard region" (high beta, low delta) in
    # training batches. Off by default, same as the Python version.
    hard_region_prob::Float64 = 0.0
    hard_beta_low_norm::Float64 = 0.85
    hard_delta_high_norm::Float64 = 0.20
end

"""
    a_support_from_shock_params(rho, sigma_eps, a_sigma_mult, a_ss=1.0)

Physical productivity support `[A_low, A_high] = exp(± n * sigma_stat) * A_ss`,
`sigma_stat = sigma_eps / sqrt(1 - rho^2)`. Used for both the NN state
normalization and the TI spline grid, so the two solvers share the same support.
"""
function a_support_from_shock_params(rho::Real, sigma_eps::Real, a_sigma_mult::Real, a_ss::Real=1.0)
    one_m = max(1e-4, 1.0 - rho^2)
    sigma_stat = sigma_eps / sqrt(one_m)
    w = a_sigma_mult * sigma_stat
    a_low = exp(-w) * a_ss
    a_high = exp(w) * a_ss
    return a_low, max(a_high, a_low + 1e-6)
end

"""
    steady_state(p::RBCParams)

Scalar steady state at `A = 1`. Returns `(k_ss, c_ss, y_ss, A_ss)`.
"""
function steady_state(p::RBCParams)
    alpha, beta, delta = p.alpha, p.beta, p.delta
    A_ss = 1.0
    term = (1.0 / beta - (1.0 - delta)) / (alpha * A_ss)
    k_ss = term^(1.0 / (alpha - 1.0))
    y_ss = A_ss * k_ss^alpha
    c_ss = y_ss - delta * k_ss
    return k_ss, c_ss, y_ss, A_ss
end

"""
    steady_state_batch(alpha, beta, delta)

Elementwise (A = 1) steady state; works for scalars or arrays/broadcastable
inputs alike, replacing the separate scalar/tensor implementations in Python.
"""
function steady_state_batch(alpha, beta, delta)
    term = @. (1.0 / beta - (1.0 - delta)) / alpha
    k_ss = @. term^(1.0 / (alpha - 1.0))
    y_ss = @. k_ss^alpha
    c_ss = @. y_ss - delta * k_ss
    return k_ss, c_ss, y_ss
end

"""
    params_to_dict(p::RBCParams) -> Dict{String,Any}

Plain, JSON-friendly representation of `p` (tuples become vectors), mirroring
`dataclasses.asdict(params)` in the Python version.
"""
function params_to_dict(p::RBCParams)
    d = Dict{String,Any}()
    for f in fieldnames(RBCParams)
        v = getfield(p, f)
        d[string(f)] = v isa Tuple ? collect(v) : v
    end
    return d
end

"""
    params_from_dict(d) -> RBCParams

Inverse of [`params_to_dict`](@ref); accepts `Dict{String,<:Any}` (e.g. from
`JSON.parsefile` or a loaded BSON dict-of-dicts).
"""
function params_from_dict(d)
    kwargs = Dict{Symbol,Any}()
    for f in fieldnames(RBCParams)
        key = string(f)
        haskey(d, key) || continue
        v = d[key]
        kwargs[f] = v isa AbstractVector ? Tuple(Float64.(v)) : v
    end
    return RBCParams(; kwargs...)
end

"""
    sample_params_uniform(base::RBCParams, rng::AbstractRNG) -> RBCParams

Draw one structural parameter set uniformly from `base`'s training bounds,
keeping all bounds/config fields fixed. Mirrors
`RBCSolver._sample_params_numpy` / `diagnose_divergence.sample_params`.
"""
function sample_params_uniform(base::RBCParams, rng::AbstractRNG)
    unif(b) = b[1] + rand(rng) * (b[2] - b[1])
    return RBCParams(
        alpha=unif(base.alpha_bounds), beta=unif(base.beta_bounds), delta=unif(base.delta_bounds),
        gamma=unif(base.gamma_bounds), rho=unif(base.rho_bounds), sigma_eps=unif(base.sigma_eps_bounds),
        k_bounds=base.k_bounds, A_sigma_mult=base.A_sigma_mult,
        alpha_bounds=base.alpha_bounds, beta_bounds=base.beta_bounds, delta_bounds=base.delta_bounds,
        rho_bounds=base.rho_bounds, gamma_bounds=base.gamma_bounds, sigma_eps_bounds=base.sigma_eps_bounds,
        hard_region_prob=base.hard_region_prob, hard_beta_low_norm=base.hard_beta_low_norm,
        hard_delta_high_norm=base.hard_delta_high_norm,
    )
end
