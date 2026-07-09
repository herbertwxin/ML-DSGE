# Time Iteration (TI) benchmark using cubic splines.
# Julia port of `rbc_TimeIter.py::RBCTISolver`, with one robustness fix on top
# (see `TISolver`/`solve!` docs): the iterated/interpolated policy is a
# *bounded consumption share* rather than a raw consumption level.

"""
    TISolver

Time-Iteration solver over a `(k, A)` grid, using cubic B-spline interpolation
(Interpolations.jl) of the consumption **share** `frac ∈ (0,1)` — not the raw
consumption level — with linear extrapolation outside the grid.

### Why a share, not a level (fix vs. a literal port of `rbc_TimeIter.py`)

The Python version iterates directly on the consumption *level* `c(k, A)`.
That representation has an unstable degenerate fixed point: if `c` dips near
zero anywhere the interpolant is evaluated (e.g. transient Runge-type
oscillation of the cubic spline during early iterations, more likely for
volatile/highly-persistent calibrations), `mu' = c'^(-gamma)` explodes at that
point, which pulls the *whole* grid's `c_target = (beta*E[mu'*R'])^(-1/gamma)`
toward zero on the next sweep — a self-reinforcing collapse to `c ≈ 0`
everywhere, which is a spurious (economically nonsensical) fixed point of the
undamped map. This was observed in practice: for some calibrations near the
edges of the training bounds, `solve!` on a raw level would converge
(`diff < tol`) to a policy grid of order `1e-6`–`1e-3` almost everywhere,
instead of order `c_ss`.

Iterating on the **share** `frac = c / resources` instead avoids this: `frac`
is clamped to `(frac_floor, 1-frac_floor)` at every use (both when it's read
back to reconstruct `c_prime` for the expectation, and when the new target is
computed), so no single quadrature node's numerics can ever push the *whole*
grid toward a degenerate corner. This is also exactly the representation the
NN uses (a sigmoid-bounded share), so both solvers share the same numerical
character, and it makes [`simulate`](@ref)'s boundary extrapolation simpler
and safe by construction (see below).
"""
mutable struct TISolver
    p::RBCParams
    k_ss::Float64
    c_ss::Float64
    y_ss::Float64
    A_ss::Float64
    n_k::Int
    n_A::Int
    k_min::Float64
    k_max::Float64
    A_min::Float64
    A_max::Float64
    k_nodes::AbstractRange{Float64}
    A_nodes::AbstractRange{Float64}
    frac_policy::Matrix{Float64}
    z_nodes::Vector{Float64}
    z_weights::Vector{Float64}
end

function TISolver(p::RBCParams)
    k_ss, c_ss, y_ss, A_ss = steady_state(p)

    n_k, n_A = 30, 15
    k_min = p.k_bounds[1] * k_ss
    k_max = p.k_bounds[2] * k_ss
    A_min, A_max = a_support_from_shock_params(p.rho, p.sigma_eps, p.A_sigma_mult, A_ss)
    @info "TI productivity grid" A_min A_max

    k_nodes = range(k_min, k_max; length=n_k)
    A_nodes = range(A_min, A_max; length=n_A)

    # Initial guess: consumption share at the (A=1) steady state everywhere,
    # mirroring how the NN's output bias is initialized (learn_rbc.py parity).
    res_ss = y_ss + (1.0 - p.delta) * k_ss
    frac_ss = c_ss / res_ss
    frac_policy = fill(frac_ss, n_k, n_A)

    n_quad = 7
    nodes, weights = gausshermite(n_quad)
    z_nodes = nodes .* sqrt(2)
    z_weights = weights ./ sqrt(pi)

    return TISolver(p, k_ss, c_ss, y_ss, A_ss, n_k, n_A, k_min, k_max, A_min, A_max,
                     k_nodes, A_nodes, frac_policy, z_nodes, z_weights)
end

_spline(ti::TISolver) = extrapolate(
    scale(interpolate(ti.frac_policy, BSpline(Cubic(Line(OnGrid())))), ti.k_nodes, ti.A_nodes),
    Line(),
)

"""
    solve!(ti::TISolver; tol=1e-6, max_iter=1000, damping=0.5, frac_floor=1e-6)

Fixed-point (time) iteration on the consumption *share* at the grid nodes
(see [`TISolver`](@ref) for why a share rather than a level). Returns the
final interpolant (a callable `itp(k, A) -> frac`).
"""
function solve!(ti::TISolver; tol::Float64=1e-6, max_iter::Int=1000, damping::Float64=0.5,
                 frac_floor::Float64=1e-6)
    p = ti.p
    itp = _spline(ti)
    for iter in 1:max_iter
        itp = _spline(ti)
        frac_new = similar(ti.frac_policy)
        for i in 1:ti.n_k, j in 1:ti.n_A
            k = ti.k_nodes[i]
            A = ti.A_nodes[j]

            y = A * k^p.alpha
            res = y + (1 - p.delta) * k

            frac_old = clamp(ti.frac_policy[i, j], frac_floor, 1.0 - frac_floor)
            c_old = frac_old * res
            k_prime = res - c_old
            k_prime_c = clamp(k_prime, ti.k_min, ti.k_max)

            E_rhs = 0.0
            for z in eachindex(ti.z_nodes)
                eps = ti.z_nodes[z]
                logA_prime = p.rho * log(A) + p.sigma_eps * eps
                A_prime = exp(logA_prime)
                A_prime_c = clamp(A_prime, ti.A_min, ti.A_max)

                frac_prime = clamp(itp(k_prime_c, A_prime_c), frac_floor, 1.0 - frac_floor)
                resources_prime = A_prime_c * k_prime_c^p.alpha + (1 - p.delta) * k_prime_c
                c_prime = frac_prime * resources_prime
                mu_prime = c_prime^(-p.gamma)
                R_prime = p.alpha * A_prime * k_prime_c^(p.alpha - 1) + (1 - p.delta)
                E_rhs += ti.z_weights[z] * mu_prime * R_prime
            end

            c_target = (p.beta * E_rhs)^(-1 / p.gamma)
            frac_target = clamp(c_target / res, frac_floor, 1.0 - frac_floor)
            frac_new[i, j] = damping * frac_target + (1 - damping) * frac_old
        end

        diff = maximum(abs.(frac_new .- ti.frac_policy))
        ti.frac_policy = frac_new
        if iter % 10 == 0
            @info "TI iteration" iter diff
        end
        if diff < tol
            @info "TI converged" iter diff
            return _spline(ti)
        end
        if iter == max_iter
            @warn "TI did not converge" diff
        end
    end
    return _spline(ti)
end

"""
    simulate(ti::TISolver, policy; T=200, k0=nothing, A0=nothing, rng=Random.default_rng())

Simulate the TI-solved economy for `T` periods given an interpolant `policy`
(as returned by [`solve!`](@ref); note `policy(k, A)` returns a consumption
*share*, not a level).

The policy is looked up at `(k, A)` clamped to the fitted grid (so we never
ask the spline to extrapolate), then the resulting share is applied to the
*true, unclamped* resources at `(kt, At)` — since the policy is already a
bounded share, this is a safe, well-defined extrapolation rule by
construction (no division by a possibly-tiny boundary resources term is
needed, unlike a raw consumption-level policy). The *physical* state series
returned is never clamped, so `k`/`A` out-of-bounds diagnostics computed from
it stay meaningful.
"""
function simulate(ti::TISolver, policy; T::Int=200, k0=nothing, A0=nothing,
                   rng::AbstractRNG=Random.default_rng())
    p = ti.p
    k0 = k0 === nothing ? ti.k_ss : k0
    A0 = A0 === nothing ? ti.A_ss : A0

    k = zeros(T + 1)
    A = zeros(T + 1)
    c = zeros(T)
    y = zeros(T)
    inv = zeros(T)
    k[1] = k0
    A[1] = A0
    eps = randn(rng, T)

    for t in 1:T
        kt, At = k[t], A[t]
        yt = At * kt^p.alpha
        y[t] = yt
        resources = yt + (1 - p.delta) * kt

        kt_q = clamp(kt, ti.k_min, ti.k_max)
        At_q = clamp(At, ti.A_min, ti.A_max)
        frac = clamp(policy(kt_q, At_q), 1e-6, 1.0 - 1e-6)
        ct = frac * resources
        c[t] = ct

        inv[t] = yt - ct
        k[t+1] = resources - ct

        logA_next = p.rho * log(max(At, 1e-8)) + p.sigma_eps * eps[t]
        A[t+1] = exp(logA_next)
    end

    return (capital=k[1:T], productivity=A[1:T], consumption=c, output=y, investment=inv)
end
