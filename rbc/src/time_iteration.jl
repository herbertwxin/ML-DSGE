# Time Iteration benchmark: Coleman operator with an exact per-node Euler solve.
#
# Earlier versions of this file (and the Python original, `rbc_TimeIter.py`)
# used naive successive approximation: evaluate the Euler right-hand side at
# the next-period capital implied by the *previous* policy, invert marginal
# utility, and damp. That map is not a contraction — for calibrations near the
# training-bound edges (high beta, low delta, high rho) it oscillated below
# tolerance forever, and for volatile draws it collapsed to a spurious
# `c ≈ 0` fixed point. See README "Fix applied: proper Coleman time iteration".
#
# The rewrite applies the textbook Coleman operator: at every grid node,
# *solve* the Euler equation
#
#     u'(c) = beta * E[ u'(c'(k', A')) * R(k', A') ],   k' = resources - c
#
# for `c` with the future policy held fixed. The left side is strictly
# decreasing in `c` and the right side strictly increasing (higher `c` means
# lower `k'`, hence higher `R` and lower `c'`), so the gap function has a
# unique root that bisection brackets by construction. Under standard
# monotonicity/concavity conditions this operator is a contraction, so the
# outer loop converges without damping.

"""
    TISolver(p::RBCParams; n_k=30, n_A=15, n_quad=7)

Grid, quadrature, and calibration for the Time-Iteration benchmark. The
capital grid spans `p.k_bounds .* k_ss(A=1)`; the productivity grid spans the
same `±A_sigma_mult * sigma_stat` box the NN normalizes on. Immutable — call
[`solve`](@ref) to produce a [`TIPolicy`](@ref).
"""
struct TISolver{KR<:AbstractRange{Float64},AR<:AbstractRange{Float64}}
    p::RBCParams
    k_nodes::KR
    A_nodes::AR
    quad::Quadrature
end

function TISolver(p::RBCParams; n_k::Int=30, n_A::Int=15, n_quad::Int=7)
    k_min, k_max = k_support(p)
    A_min, A_max = a_support_from_shock_params(p.rho, p.sigma_eps, p.A_sigma_mult)
    return TISolver(p, range(k_min, k_max; length=n_k),
                    range(A_min, A_max; length=n_A), Quadrature(n_quad))
end

"""
    TIPolicy

Converged consumption-share policy on the TI grid: a cubic B-spline
interpolant of `frac = c / resources` plus solve metadata. Callable and
usable with the shared [`simulate`](@ref) via [`consumption_share`](@ref).

The interpolated object is a *share* in `(0, 1)`, not a consumption level:
a share stays economically meaningful when the simulated state leaves the
fitted grid (apply the boundary share to the true resources), whereas
extrapolating a raw consumption level can violate the budget constraint and
explode.
"""
struct TIPolicy{I}
    p::RBCParams
    itp::I
    frac::Matrix{Float64}
    k_min::Float64
    k_max::Float64
    A_min::Float64
    A_max::Float64
    frac_floor::Float64
    converged::Bool
    iterations::Int
    residual::Float64
end

"""
    consumption_share(pol::TIPolicy, k, A) -> frac in (0, 1)

Evaluate the policy at `(k, A)`. Queries are clamped to the fitted grid box
and the share to `(frac_floor, 1 - frac_floor)`; apply the share to the true
resources at the *unclamped* state to get consumption.
"""
function consumption_share(pol::TIPolicy, k::Real, A::Real)
    frac = pol.itp(clamp(k, pol.k_min, pol.k_max), clamp(A, pol.A_min, pol.A_max))
    return clamp(frac, pol.frac_floor, 1.0 - pol.frac_floor)
end

(pol::TIPolicy)(k::Real, A::Real) = consumption_share(pol, k, A)

share_spline(ti::TISolver, frac::Matrix{Float64}) =
    scale(interpolate(frac, BSpline(Cubic(Line(OnGrid())))), ti.k_nodes, ti.A_nodes)

"""
    solve(ti::TISolver; tol=1e-7, max_iter=2000, frac_floor=1e-6, verbose=false)

Run Coleman time iteration to a fixed point of the consumption-share policy.
`tol` is the sup-norm of the share update (shares are O(0.1–0.5), so `1e-7`
is a relative accuracy of ~1e-6). Returns a [`TIPolicy`](@ref); check
`.converged` when running near the edges of the parameter box.
"""
function solve(ti::TISolver; tol::Float64=1e-7, max_iter::Int=2000,
               frac_floor::Float64=1e-6, verbose::Bool=false)
    frac = fill(steady_state_share(ti.p), length(ti.k_nodes), length(ti.A_nodes))
    frac_next = similar(frac)
    diff = Inf
    iter = 0
    while iter < max_iter
        iter += 1
        coleman_step!(frac_next, ti, share_spline(ti, frac), frac_floor)
        diff = maximum(abs(a - b) for (a, b) in zip(frac_next, frac))
        frac, frac_next = frac_next, frac
        verbose && iter % 50 == 0 && @info "TI sweep" iter diff
        diff < tol && break
    end
    converged = diff < tol
    if converged
        verbose && @info "TI converged" iter diff
    else
        @warn "Time iteration did not converge" ti.p.beta ti.p.delta ti.p.rho iter diff
    end
    return TIPolicy(ti.p, share_spline(ti, frac), frac,
                    first(ti.k_nodes), last(ti.k_nodes), first(ti.A_nodes), last(ti.A_nodes),
                    frac_floor, converged, iter, diff)
end

"""
    coleman_step!(frac_next, ti, itp, frac_floor)

One sweep of the Coleman operator: solve the Euler equation exactly at every
grid node, holding the future policy `itp` fixed. Nodes are independent, so
the sweep is threaded.
"""
function coleman_step!(frac_next::Matrix{Float64}, ti::TISolver, itp, frac_floor::Float64)
    p = ti.p
    Threads.@threads for j in eachindex(ti.A_nodes)
        A = ti.A_nodes[j]
        for (i, k) in enumerate(ti.k_nodes)
            res = resources(k, A, p.alpha, p.delta)
            frac_next[i, j] = solve_node(ti, itp, res, A, frac_floor)
        end
    end
    return frac_next
end

"""
    euler_gap(ti, itp, c, res, A, frac_floor)

`u'(c) - beta * E[u'(c') R']` at a node with resources `res` and productivity
`A`, where `k' = res - c` and next-period consumption comes from the share
policy `itp` (queried inside the grid box, applied to true resources at the
actual `(k', A')`). Strictly decreasing in `c`.
"""
function euler_gap(ti::TISolver, itp, c::Float64, res::Float64, A::Float64, frac_floor::Float64)
    p = ti.p
    k1 = res - c
    k1_q = clamp(k1, first(ti.k_nodes), last(ti.k_nodes))
    A_lo, A_hi = first(ti.A_nodes), last(ti.A_nodes)
    logA = log(A)
    rhs = 0.0
    for (z, w) in zip(ti.quad.nodes, ti.quad.weights)
        A1 = exp(p.rho * logA + p.sigma_eps * z)
        frac1 = clamp(itp(k1_q, clamp(A1, A_lo, A_hi)), frac_floor, 1.0 - frac_floor)
        c1 = frac1 * resources(k1, A1, p.alpha, p.delta)
        rhs += w * marginal_utility(c1, p.gamma) * gross_return(k1, A1, p.alpha, p.delta)
    end
    return marginal_utility(c, p.gamma) - p.beta * rhs
end

"""
    solve_node(ti, itp, res, A, frac_floor) -> frac

Solve the node's Euler equation for consumption by bisection on
`c ∈ (frac_floor, 1 - frac_floor) * res`. The bracket is guaranteed:
`euler_gap → +∞` as `c → 0` (marginal utility blows up) and `→ -∞` as
`c → res` (the return on vanishing capital blows up). Corner cases where the
gap does not change sign on the interval return the corresponding bound.
"""
function solve_node(ti::TISolver, itp, res::Float64, A::Float64, frac_floor::Float64)
    lo = frac_floor * res
    hi = (1.0 - frac_floor) * res
    euler_gap(ti, itp, lo, res, A, frac_floor) <= 0.0 && return frac_floor
    euler_gap(ti, itp, hi, res, A, frac_floor) >= 0.0 && return 1.0 - frac_floor
    while hi - lo > 1e-12 * res
        mid = 0.5 * (lo + hi)
        if euler_gap(ti, itp, mid, res, A, frac_floor) > 0.0
            lo = mid
        else
            hi = mid
        end
    end
    return clamp(0.5 * (lo + hi) / res, frac_floor, 1.0 - frac_floor)
end
