# RBC with endogenous labor — the second model on the engine, fully contained
# in this file (economics, engine-interface methods, NN policy wrapper, a
# labor-augmented Coleman TI benchmark, simulate loop, validation panel).
#
# Preferences (CRRA consumption, separable isoelastic labor disutility):
#
#     u(c, n) = c^(1-gamma)/(1-gamma) - chi * n^(1+1/nu)/(1+1/nu)
#
# Technology: y = A k^alpha n^(1-alpha);  resources = y + (1-delta) k.
# Two optimality conditions:
#
#   intratemporal:  chi n^(1/nu) = c^(-gamma) (1-alpha) A k^alpha n^(-alpha)
#   Euler:          c^(-gamma)   = beta E[c'^(-gamma) R'],
#                   R' = alpha A' k'^(alpha-1) n'^(1-alpha) + 1 - delta
#
# SUBSTITUTED FORMULATION (see comment/some_math.tm): the intratemporal
# condition is a bijection between c and n given (k, A) — imposing it
# exactly and substituting into the Euler equation leaves ONE functional
# equation in one control. The network outputs hours n (naturally bounded in
# (0, 1)), consumption follows in closed form,
#
#     c(n) = ((1-alpha) A k^alpha / (chi n^(alpha + 1/nu)))^(1/gamma),
#
# and the training loss is the single substituted Euler residual plus the
# over-saving penalty (the FOCs are first-order only; transversality still
# needs the state box to be invariant). The TI benchmark solves the same
# substituted equation from the other direction — a bisection on c with
# n(c) = ((1-alpha) A k^alpha c^(-gamma) / chi)^(nu/(1 + alpha nu)) — so
# both solvers discretize the identical one-dimensional problem.
#
# `nu` (Frisch elasticity) is a sampled structural parameter; `chi` is NOT —
# it is pinned per calibration so steady-state hours equal `n_ss_target`
# (the standard normalization), which keeps the steady state analytic:
# with kappa = k/n = ((1/beta - (1-delta))/alpha)^(1/(alpha-1)) from the
# Euler equation, k_ss = kappa * n_ss and chi follows from the intratemporal
# condition at the steady state.

"""
    RBCLaborParams

Structural parameters plus training bounds for the RBC model with labor
choice. Same layout conventions as `RBCParams`: scalar calibration fields,
`*_bounds` for the training box, sampling config at the end. `n_ss_target`
pins the labor-disutility weight `chi` per calibration and is never sampled.
"""
Base.@kwdef struct RBCLaborParams
    alpha::Float64 = 0.30      # capital share
    beta::Float64  = 0.95      # discount factor
    delta::Float64 = 0.1       # depreciation rate
    gamma::Float64 = 2.0       # risk aversion (CRRA, consumption)
    rho::Float64   = 0.90      # persistence of productivity shock
    sigma_eps::Float64 = 0.02  # std dev of shock innovation
    nu::Float64    = 1.0       # Frisch elasticity of labor supply
    n_ss_target::Float64 = 1/3 # steady-state hours (pins chi; not sampled)

    k_bounds::Tuple{Float64,Float64} = (0.5, 1.5)
    A_sigma_mult::Float64 = 3.0

    alpha_bounds::Tuple{Float64,Float64} = (0.20, 0.45)
    beta_bounds::Tuple{Float64,Float64}  = (0.90, 0.99)
    delta_bounds::Tuple{Float64,Float64} = (0.02, 0.15)
    rho_bounds::Tuple{Float64,Float64}   = (0.85, 0.99)
    gamma_bounds::Tuple{Float64,Float64} = (0.5, 4.0)
    sigma_eps_bounds::Tuple{Float64,Float64} = (0.005, 0.05)
    nu_bounds::Tuple{Float64,Float64}    = (0.5, 2.0)

    hard_region_prob::Float64 = 0.5
    hard_beta_low_norm::Float64 = 0.85
    hard_delta_high_norm::Float64 = 0.20
end

# ---------------------------------------------------------------------------
# Steady state and supports (analytic thanks to the n_ss normalization)
# ---------------------------------------------------------------------------

"Capital-labor ratio `kappa = k/n` from the steady-state Euler equation (A=1)."
steady_state_kappa(alpha, beta, delta) = @. ((1 / beta - (1 - delta)) / alpha)^(1 / (alpha - 1))

"""
    labor_chi(alpha, beta, delta, gamma, nu, n_ss)

Labor-disutility weight making steady-state hours equal `n_ss`:
`chi = w_ss * c_ss^(-gamma) / n_ss^(1/nu)` with `w_ss = (1-alpha) kappa^alpha`
and `c_ss = (kappa^alpha - delta kappa) n_ss`. Broadcastable (Float32 batches).
"""
function labor_chi(alpha, beta, delta, gamma, nu, n_ss)
    kappa = steady_state_kappa(alpha, beta, delta)
    return @. (1 - alpha) * kappa^alpha * ((kappa^alpha - delta * kappa) * n_ss)^(-gamma) * n_ss^(-(1 / nu))
end

"""
    steady_state(p::RBCLaborParams) -> (k, c, y, n, w, chi, A)

Deterministic steady state at `A = 1`, hours normalized to `n_ss_target`.
"""
function steady_state(p::RBCLaborParams)
    kappa = steady_state_kappa(p.alpha, p.beta, p.delta)
    n = p.n_ss_target
    k = kappa * n
    y = kappa^p.alpha * n
    c = y - p.delta * k
    w = (1 - p.alpha) * kappa^p.alpha
    chi = w * c^(-p.gamma) / n^(1 / p.nu)
    return (k=k, c=c, y=y, n=n, w=w, chi=chi, A=1.0)
end

"Steady-state consumption share of total resources."
function steady_state_share(p::RBCLaborParams)
    ss = steady_state(p)
    return ss.c / (ss.y + (1 - p.delta) * ss.k)
end

"Capital support `k_bounds .* k_ss(A=1)` with `k_ss = kappa * n_ss_target`."
function k_support(p::RBCLaborParams)
    k_ss = steady_state_kappa(p.alpha, p.beta, p.delta) * p.n_ss_target
    return p.k_bounds[1] * k_ss, p.k_bounds[2] * k_ss
end

# ---------------------------------------------------------------------------
# Parameter sampling (7 structural parameters; chi is implied, never sampled)
# ---------------------------------------------------------------------------

"Copy of `base` with the structural scalars in `structural` replaced."
function with_calibration(base::RBCLaborParams, structural::NamedTuple)
    fields = (f => get(structural, f, getfield(base, f)) for f in fieldnames(RBCLaborParams))
    return RBCLaborParams(; fields...)
end

"Draw one structural parameter set uniformly from `base`'s training bounds."
function sample_params_uniform(base::RBCLaborParams, rng::AbstractRNG)
    unif((lo, hi)) = lo + rand(rng) * (hi - lo)
    structural = (
        alpha=unif(base.alpha_bounds), beta=unif(base.beta_bounds), delta=unif(base.delta_bounds),
        gamma=unif(base.gamma_bounds), rho=unif(base.rho_bounds),
        sigma_eps=unif(base.sigma_eps_bounds), nu=unif(base.nu_bounds),
    )
    return with_calibration(base, structural)
end

# ===========================================================================
# Engine interface: network spec, batch sampling, two-residual training loss
# ===========================================================================

"""
    policy_spec(p::RBCLaborParams)

Single-output policy network: a normalized 9-vector
`(k, A, alpha, beta, delta, rho, gamma, sigma_eps, nu)` in, hours
`n in (0, 1)` (sigmoid) out — consumption is not learned, it follows from
the intratemporal condition in closed form (see the header). Output bias
pre-set so the untrained policy starts at `n_ss_target`.
"""
function policy_spec(p::RBCLaborParams)
    n_ss = p.n_ss_target
    return (input_dim=9, hidden=[64, 64, 64, 64], output_dim=1,
            output_bias=log(n_ss / (1 - n_ss)))
end

"""
    sample_batch(p::RBCLaborParams, rng, batch_size)

Float32 batch over the normalized state × parameter box (9 rows), with the
same hard-corner oversampling as the baseline RBC model (high beta, low
delta, edge gamma). Also returns the implied per-draw `chi` — exogenous to
the network, so it stays off the AD tape.
"""
function sample_batch(p::RBCLaborParams, rng::AbstractRNG, batch_size::Int)
    # Rows: k, A, alpha, beta, delta, rho, gamma, sigma_eps, nu (all in [0, 1]).
    inputs = rand(rng, Float32, 9, batch_size)

    if p.hard_region_prob > 0.0
        beta_low = Float32(p.hard_beta_low_norm)
        delta_high = Float32(p.hard_delta_high_norm)
        gamma_width = 0.20f0
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

    b32(bounds) = Float32.(bounds)
    alpha     = denormalize01(inputs[3, :], b32(p.alpha_bounds)...)
    beta      = denormalize01(inputs[4, :], b32(p.beta_bounds)...)
    delta     = denormalize01(inputs[5, :], b32(p.delta_bounds)...)
    rho       = denormalize01(inputs[6, :], b32(p.rho_bounds)...)
    gamma     = denormalize01(inputs[7, :], b32(p.gamma_bounds)...)
    sigma_eps = denormalize01(inputs[8, :], b32(p.sigma_eps_bounds)...)
    nu        = denormalize01(inputs[9, :], b32(p.nu_bounds)...)

    A_mult = Float32(p.A_sigma_mult)
    sigma_stat = @. sigma_eps / sqrt(max(1f-4, 1 - rho^2))
    A_low = @. exp(-A_mult * sigma_stat)
    A_high = @. max(exp(A_mult * sigma_stat), A_low + 1f-6)

    n_ss = Float32(p.n_ss_target)
    k_ss = steady_state_kappa(alpha, beta, delta) .* n_ss
    chi = labor_chi(alpha, beta, delta, gamma, nu, n_ss)
    k_low = Float32(p.k_bounds[1]) .* k_ss
    k_high = Float32(p.k_bounds[2]) .* k_ss

    k = denormalize01(inputs[1, :], k_low, k_high)
    A = denormalize01(inputs[2, :], A_low, A_high)

    return (inputs=inputs, k=k, A=A, k_low=k_low, k_high=k_high, A_low=A_low, A_high=A_high,
            alpha=alpha, beta=beta, delta=delta, rho=rho, gamma=gamma, sigma_eps=sigma_eps,
            nu=nu, chi=chi)
end

"""
    labor_terms(model, batch, quad::Quadrature) -> (euler, k_oob)

Per-sample ingredients of the substituted training loss. The network's hours
output is floored at 0.01 (1% of the time endowment): `c(n)` blows up as
`n -> 0`, and the floor keeps the Float32 powers finite without restricting
any economically relevant policy.

- `euler`: normalized substituted Euler residual
  `(beta E[u_c' R'] - u_c) / u_c`, where consumption today AND at every
  quadrature node tomorrow comes from the intratemporal condition
  `c = ((1-alpha) A k^alpha / (chi n^(alpha+1/nu)))^(1/gamma)` — the
  intratemporal condition holds exactly by construction, so it contributes
  no residual. Expectations by Gauss-Hermite quadrature with all nodes
  through the network in one call (9×(B*Q) input, one output row).

  Feasibility cap: `c` is capped at `0.99 * resources`. Without it, hours
  choices whose intratemporal-consistent consumption exceeds resources send
  `k'` to the clamp floor and `1/u_c` to extreme values, blowing up the
  normalized residual (observed ~1e12 at random init). The cap keeps `k'`
  bounded away from zero, binds only for infeasible off-equilibrium
  policies, and is inactive at the solution — where the intratemporal
  condition therefore holds exactly.
- `k_oob`: over-saving penalty on normalized `k'` outside `[0, 1]`, identical
  in role to the baseline model. It also disciplines infeasible hours: too
  little labor makes capped consumption pin `k'` at `0.01 * resources`,
  which lies below the capital box.
"""
function labor_terms(model, batch, quad::Quadrature)
    (; inputs, k, A, k_low, k_high, A_low, A_high,
       alpha, beta, delta, rho, gamma, sigma_eps, nu, chi) = batch

    n = max.(vec(model(inputs)), 1f-2)        # hours; c(n) explodes as n -> 0

    res = @. A * k^alpha * n^(1 - alpha) + (1 - delta) * k
    c = min.((@. ((1 - alpha) * A * k^alpha / (chi * n^(alpha + 1 / nu)))^(1 / gamma)),
             0.99f0 .* res)
    k1 = max.(res .- c, 1f-8)
    mu = c .^ (-gamma)

    k1_norm = normalize01(k1, k_low, k_high)
    logA = log.(A)
    theta_rows = inputs[3:9, :]

    Q = length(quad.nodes)
    zrow, wrow, onesrow = Flux.Zygote.ignore() do
        z = copyto!(similar(k, 1, Q), reshape(Float32.(quad.nodes), 1, Q))
        w = copyto!(similar(k, 1, Q), reshape(Float32.(quad.weights), 1, Q))
        o = fill!(similar(k, 1, Q), 1f0)
        (z, w, o)
    end

    A1 = @. exp(rho * logA + sigma_eps * zrow)              # B×Q
    A1_norm = @. (A1 - A_low) / (A_high - A_low)
    k1_rep = k1_norm .* onesrow
    inputs1 = vcat(reshape(k1_rep, 1, :), reshape(A1_norm, 1, :),
                   repeat(theta_rows, 1, Q))                # 9×(B*Q)
    n1 = max.(reshape(vec(model(inputs1)), :, Q), 1f-2)     # B×Q
    res1 = @. A1 * k1^alpha * n1^(1 - alpha) + (1 - delta) * k1
    c1 = min.((@. ((1 - alpha) * A1 * k1^alpha / (chi * n1^(alpha + 1 / nu)))^(1 / gamma)),
              0.99f0 .* res1)
    R1 = @. alpha * A1 * k1^(alpha - 1) * n1^(1 - alpha) + (1 - delta)
    rhs = beta .* vec(sum(@.(c1^(-gamma) * R1 * wrow); dims=2))

    k_oob = @. max(k1_norm - 1, 0)^2 + max(-k1_norm, 0)^2
    return (rhs .- mu) ./ mu, k_oob
end

"""
    labor_loss(model, batch, quad; k_oob_weight=1.0)

Training objective: mean squared normalized substituted Euler residual plus
`k_oob_weight` times the mean over-saving penalty. One equilibrium-condition
residual only — the intratemporal condition is imposed exactly through
`c(n)`, so no `intra_weight` exists to tune.
"""
function labor_loss(model, batch, quad::Quadrature; k_oob_weight::Real=1.0)
    euler, k_oob = labor_terms(model, batch, quad)
    return mean(abs2, euler) + Float32(k_oob_weight) * mean(k_oob)
end

"The labor-model objective handed to the engine; see [`labor_loss`](@ref)."
training_loss(p::RBCLaborParams, model, batch, quad::Quadrature; k_oob_weight::Real=1.0) =
    labor_loss(model, batch, quad; k_oob_weight)

"Validation components: penalized loss, substituted-Euler MSE, penalty."
function validation_report(p::RBCLaborParams, model, batch, quad::Quadrature;
                           k_oob_weight::Real=1.0)
    euler, k_oob = labor_terms(model, batch, quad)
    euler_mse = mean(abs2, euler)
    oob = mean(k_oob)
    return (loss=euler_mse + k_oob_weight * oob, euler_mse=euler_mse, k_oob=oob)
end

# ---------------------------------------------------------------------------
# Fixed-calibration policies: `controls(policy, k, A) -> (frac, n)`
# ---------------------------------------------------------------------------

"""
    NNLaborPolicy(solver::NNSolver{RBCLaborParams}, p::RBCLaborParams)

The hours network at one fixed calibration, queried pointwise with
[`controls`](@ref) (CPU copy of the model, Float32 only at the input).
Consumption is recovered from the intratemporal condition, so the policy
carries the calibration and its implied `chi`.
"""
struct NNLaborPolicy{M}
    model::M
    p::RBCLaborParams
    chi::Float64
    theta_norm::NTuple{7,Float64}
    k_low::Float64
    k_high::Float64
    A_low::Float64
    A_high::Float64
end

function NNLaborPolicy(solver::NNSolver{RBCLaborParams}, p::RBCLaborParams)
    base = solver.p
    k_low, k_high = k_support(p)
    A_low, A_high = a_support_from_shock_params(p.rho, p.sigma_eps, base.A_sigma_mult)
    theta = (normalize_scalar(p.alpha, base.alpha_bounds),
             normalize_scalar(p.beta, base.beta_bounds),
             normalize_scalar(p.delta, base.delta_bounds),
             normalize_scalar(p.rho, base.rho_bounds),
             normalize_scalar(p.gamma, base.gamma_bounds),
             normalize_scalar(p.sigma_eps, base.sigma_eps_bounds),
             normalize_scalar(p.nu, base.nu_bounds))
    return NNLaborPolicy(cpu(solver.model), p, steady_state(p).chi,
                         theta, k_low, k_high, A_low, A_high)
end

"""
    controls(policy, k, A) -> (frac, n)

Consumption share and hours at state `(k, A)`; the labor-model analogue of
`consumption_share`. Implemented by [`NNLaborPolicy`](@ref) and
[`LaborTIPolicy`](@ref), so one `simulate` drives both.

For the NN policy, hours come from the network and consumption from the
intratemporal condition; the returned share `c/res` is clamped below 1, which
keeps `k' > 0` in simulation (the clamp can only bind far off equilibrium,
where intratemporal-consistent consumption would exceed resources).
"""
function controls(pol::NNLaborPolicy, k::Real, A::Real)
    kn = (k - pol.k_low) / (pol.k_high - pol.k_low)
    an = (A - pol.A_low) / (pol.A_high - pol.A_low)
    x = Float32[kn, an, pol.theta_norm...]
    p = pol.p
    n = clamp(Float64(pol.model(reshape(x, 9, 1))[1]), 1e-2, 1.0)
    c = ((1 - p.alpha) * A * k^p.alpha / (pol.chi * n^(p.alpha + 1 / p.nu)))^(1 / p.gamma)
    res = A * k^p.alpha * n^(1 - p.alpha) + (1 - p.delta) * k
    return clamp(c / res, 1e-6, 1.0 - 1e-6), n
end

"Simulate the trained labor-model network at calibration `p`."
simulate(solver::NNSolver{RBCLaborParams}, p::RBCLaborParams; kwargs...) =
    simulate(NNLaborPolicy(solver, p), p; kwargs...)

"""
    simulate(policy, p::RBCLaborParams; T=200, k0=nothing, A0=nothing, rng)

Simulate under any policy implementing [`controls`](@ref). Same conventions
as the baseline model (Float64 states, share keeps `k' > 0`, state never
clamped); returns `(capital, productivity, consumption, output, investment,
hours)`.
"""
function simulate(policy, p::RBCLaborParams; T::Int=200, k0=nothing, A0=nothing,
                  rng::AbstractRNG=Random.default_rng())
    ss = steady_state(p)
    kt = something(k0, ss.k)
    At = something(A0, ss.A)

    k = Vector{Float64}(undef, T)
    A = Vector{Float64}(undef, T)
    c = Vector{Float64}(undef, T)
    y = Vector{Float64}(undef, T)
    inv = Vector{Float64}(undef, T)
    n = Vector{Float64}(undef, T)
    shocks = randn(rng, T)

    for t in 1:T
        k[t] = kt
        A[t] = At
        frac, nt = controls(policy, kt, At)
        nt = clamp(nt, 1e-6, 1.0)
        yt = At * kt^p.alpha * nt^(1 - p.alpha)
        res = yt + (1.0 - p.delta) * kt
        ct = frac * res
        y[t] = yt
        c[t] = ct
        inv[t] = yt - ct
        n[t] = nt
        kt = res - ct
        At = next_productivity(At, shocks[t], p.rho, p.sigma_eps)
    end

    return (capital=k, productivity=A, consumption=c, output=y, investment=inv, hours=n)
end

# ---------------------------------------------------------------------------
# Time-iteration benchmark with labor
#
# Coleman operator as in the baseline model, except each node carries two
# controls. The intratemporal condition gives hours in closed form given c
# (see the header), so the node solve is still a single bisection on c: the
# Euler gap u'(c) - beta E[u'(c') R'] remains strictly decreasing in c
# (higher c -> lower n -> lower resources -> lower k' -> higher R', lower c').
# ---------------------------------------------------------------------------

"""
    LaborTISolver(p::RBCLaborParams; n_k=30, n_A=15, n_quad=7)

Grids, quadrature, calibration, and the implied `chi` for the labor TI
benchmark; same `(k, A)` box conventions as the NN normalization.
"""
struct LaborTISolver{KR<:AbstractRange{Float64},AR<:AbstractRange{Float64}}
    p::RBCLaborParams
    chi::Float64
    k_nodes::KR
    A_nodes::AR
    quad::Quadrature
end

function LaborTISolver(p::RBCLaborParams; n_k::Int=30, n_A::Int=15, n_quad::Int=7)
    k_min, k_max = k_support(p)
    A_min, A_max = a_support_from_shock_params(p.rho, p.sigma_eps, p.A_sigma_mult)
    return LaborTISolver(p, steady_state(p).chi, range(k_min, k_max; length=n_k),
                         range(A_min, A_max; length=n_A), Quadrature(n_quad))
end

"""
    LaborTIPolicy

Converged `(frac, n)` policy on the TI grid: cubic B-spline interpolants of
the consumption share and hours, plus solve metadata. Queried through
[`controls`](@ref) (clamped to the grid box) and driven by the shared labor
`simulate`.
"""
struct LaborTIPolicy{IF,IN}
    p::RBCLaborParams
    itp_frac::IF
    itp_n::IN
    frac::Matrix{Float64}
    n::Matrix{Float64}
    k_min::Float64
    k_max::Float64
    A_min::Float64
    A_max::Float64
    frac_floor::Float64
    converged::Bool
    iterations::Int
    residual::Float64
end

function controls(pol::LaborTIPolicy, k::Real, A::Real)
    kq = clamp(k, pol.k_min, pol.k_max)
    Aq = clamp(A, pol.A_min, pol.A_max)
    frac = clamp(pol.itp_frac(kq, Aq), pol.frac_floor, 1.0 - pol.frac_floor)
    n = clamp(pol.itp_n(kq, Aq), 1e-6, 1.0)
    return frac, n
end

"Simulate a [`LaborTIPolicy`](@ref) at its own calibration."
simulate(pol::LaborTIPolicy; kwargs...) = simulate(pol, pol.p; kwargs...)

labor_spline(ti::LaborTISolver, m::Matrix{Float64}) =
    scale(interpolate(m, BSpline(Cubic(Line(OnGrid())))), ti.k_nodes, ti.A_nodes)

"Hours from the intratemporal condition given consumption (capped at n = 1)."
function labor_hours_given_c(ti::LaborTISolver, c::Float64, k::Float64, A::Float64)
    p = ti.p
    n = ((1 - p.alpha) * A * k^p.alpha * c^(-p.gamma) / ti.chi)^(p.nu / (1 + p.alpha * p.nu))
    return min(n, 1.0)
end

"""
    labor_euler_gap(ti, itp_frac, itp_n, c, k, A, frac_floor)

`u'(c) - beta E[u'(c') R']` at a node, with hours from the intratemporal
condition and `k' = resources(n(c)) - c`; `-Inf` when `k' <= 0` (c
infeasible), preserving the bisection bracket. Strictly decreasing in `c`.
"""
function labor_euler_gap(ti::LaborTISolver, itp_frac, itp_n, c::Float64,
                         k::Float64, A::Float64, frac_floor::Float64)
    p = ti.p
    n = labor_hours_given_c(ti, c, k, A)
    res = A * k^p.alpha * n^(1 - p.alpha) + (1 - p.delta) * k
    k1 = res - c
    k1 <= 0.0 && return -Inf
    k1_q = clamp(k1, first(ti.k_nodes), last(ti.k_nodes))
    A_lo, A_hi = first(ti.A_nodes), last(ti.A_nodes)
    logA = log(A)
    rhs = 0.0
    for (z, w) in zip(ti.quad.nodes, ti.quad.weights)
        A1 = exp(p.rho * logA + p.sigma_eps * z)
        A1_q = clamp(A1, A_lo, A_hi)
        frac1 = clamp(itp_frac(k1_q, A1_q), frac_floor, 1.0 - frac_floor)
        n1 = clamp(itp_n(k1_q, A1_q), 1e-6, 1.0)
        res1 = A1 * k1^p.alpha * n1^(1 - p.alpha) + (1 - p.delta) * k1
        c1 = frac1 * res1
        R1 = p.alpha * A1 * k1^(p.alpha - 1) * n1^(1 - p.alpha) + (1 - p.delta)
        rhs += w * c1^(-p.gamma) * R1
    end
    return c^(-p.gamma) - p.beta * rhs
end

"Consumption share and hours at the node's Euler solution `c`."
function labor_node_controls(ti::LaborTISolver, c::Float64, k::Float64, A::Float64,
                             frac_floor::Float64)
    p = ti.p
    n = labor_hours_given_c(ti, c, k, A)
    res = A * k^p.alpha * n^(1 - p.alpha) + (1 - p.delta) * k
    return clamp(c / res, frac_floor, 1.0 - frac_floor), n
end

"""
    solve_labor_node(ti, itp_frac, itp_n, k, A, frac_floor) -> (frac, n)

Bisection on `c` over `(frac_floor, 1 - frac_floor) * res_max`, where
`res_max` are resources at the hours cap `n = 1` (an upper bound on feasible
consumption). The bracket is guaranteed: the gap `-> +Inf` as `c -> 0` and is
`-Inf` once `k' <= 0`, which must happen before `c = res_max`.
"""
function solve_labor_node(ti::LaborTISolver, itp_frac, itp_n, k::Float64, A::Float64,
                          frac_floor::Float64)
    p = ti.p
    res_max = A * k^p.alpha + (1 - p.delta) * k
    lo = frac_floor * res_max
    hi = (1.0 - frac_floor) * res_max
    labor_euler_gap(ti, itp_frac, itp_n, lo, k, A, frac_floor) <= 0.0 &&
        return labor_node_controls(ti, lo, k, A, frac_floor)
    labor_euler_gap(ti, itp_frac, itp_n, hi, k, A, frac_floor) >= 0.0 &&
        return labor_node_controls(ti, hi, k, A, frac_floor)
    while hi - lo > 1e-12 * res_max
        mid = 0.5 * (lo + hi)
        if labor_euler_gap(ti, itp_frac, itp_n, mid, k, A, frac_floor) > 0.0
            lo = mid
        else
            hi = mid
        end
    end
    return labor_node_controls(ti, 0.5 * (lo + hi), k, A, frac_floor)
end

"One threaded Coleman sweep over the grid, updating both control matrices."
function coleman_labor_step!(frac_next::Matrix{Float64}, n_next::Matrix{Float64},
                             ti::LaborTISolver, itp_frac, itp_n, frac_floor::Float64)
    Threads.@threads for j in eachindex(ti.A_nodes)
        A = ti.A_nodes[j]
        for (i, k) in enumerate(ti.k_nodes)
            frac_next[i, j], n_next[i, j] = solve_labor_node(ti, itp_frac, itp_n, k, A, frac_floor)
        end
    end
    return frac_next, n_next
end

"""
    solve(ti::LaborTISolver; tol=1e-7, max_iter=2000, frac_floor=1e-6, verbose=false)

Coleman time iteration to a joint fixed point of the `(frac, n)` policy;
`tol` is the sup-norm of the update across both controls. Initialized at the
steady state. Returns a [`LaborTIPolicy`](@ref); check `.converged` near the
edges of the parameter box.
"""
function solve(ti::LaborTISolver; tol::Float64=1e-7, max_iter::Int=2000,
               frac_floor::Float64=1e-6, verbose::Bool=false)
    ss = steady_state(ti.p)
    frac = fill(steady_state_share(ti.p), length(ti.k_nodes), length(ti.A_nodes))
    n = fill(ss.n, length(ti.k_nodes), length(ti.A_nodes))
    frac_next = similar(frac)
    n_next = similar(n)
    diff = Inf
    iter = 0
    while iter < max_iter
        iter += 1
        coleman_labor_step!(frac_next, n_next, ti,
                            labor_spline(ti, frac), labor_spline(ti, n), frac_floor)
        diff = max(maximum(abs(a - b) for (a, b) in zip(frac_next, frac)),
                   maximum(abs(a - b) for (a, b) in zip(n_next, n)))
        frac, frac_next = frac_next, frac
        n, n_next = n_next, n
        verbose && iter % 50 == 0 && @info "Labor TI sweep" iter diff
        diff < tol && break
    end
    converged = diff < tol
    if converged
        verbose && @info "Labor TI converged" iter diff
    else
        @warn "Labor time iteration did not converge" ti.p.beta ti.p.delta ti.p.rho iter diff
    end
    return LaborTIPolicy(ti.p, labor_spline(ti, frac), labor_spline(ti, n), frac, n,
                         first(ti.k_nodes), last(ti.k_nodes),
                         first(ti.A_nodes), last(ti.A_nodes),
                         frac_floor, converged, iter, diff)
end

# ---------------------------------------------------------------------------
# Validation panel (NN vs labor TI, diagnostic only)
# ---------------------------------------------------------------------------

"Fixed panel of calibrations with labor-TI benchmark policies (threaded solve)."
function build_validation_panel(p::RBCLaborParams, n_cases::Int, seed::Int)
    n_cases <= 0 && return NamedTuple{(:params, :policy, :seed)}[]
    rng = Xoshiro(seed)
    cases = [sample_params_uniform(p, rng) for _ in 1:n_cases]
    panel = Vector{Any}(undef, n_cases)
    Threads.@threads for i in 1:n_cases
        panel[i] = (params=cases[i], policy=solve(LaborTISolver(cases[i])), seed=seed + i)
    end
    return panel
end

"Average NN-vs-TI gap metrics across the panel (hours included)."
function evaluate_validation_panel(p::RBCLaborParams, solver::NNSolver, panel, T::Int)
    isempty(panel) && return nothing

    metrics_list = map(panel) do item
        nn_res = simulate(solver, item.params; T, rng=Xoshiro(item.seed))
        ti_res = simulate(item.policy; T, rng=Xoshiro(item.seed))
        gap_metrics(nn_res, ti_res;
                    series=(:consumption, :capital, :output, :investment, :hours))
    end

    return (
        mean_nrmse=mean(m["aggregate"]["mean_nrmse"] for m in metrics_list),
        max_nrmse=maximum(m["aggregate"]["max_nrmse"] for m in metrics_list),
        level_ratio_c=mean(m["consumption"]["level_ratio"] for m in metrics_list),
        level_ratio_n=mean(m["hours"]["level_ratio"] for m in metrics_list),
    )
end
