# One simulation loop shared by every policy representation.
#
# A "policy" is anything implementing `consumption_share(policy, k, A)`;
# `TIPolicy` (spline) and `NNPolicy` (network at a fixed calibration) both do.
# Because every policy returns a share in (0, 1), next-period capital
# `k' = (1 - frac) * resources` is positive by construction — no clamping of
# the simulated state is needed or applied, so out-of-bounds diagnostics
# computed from the returned paths stay meaningful.

"""
    simulate(policy, p::RBCParams; T=200, k0=nothing, A0=nothing, rng=Random.default_rng())

Simulate the economy for `T` periods under `policy` at calibration `p`.
`k0`/`A0` default to the (A = 1) steady state. Returns a named tuple of
length-`T` paths `(capital, productivity, consumption, output, investment)`.

Pass an explicitly seeded `rng` (e.g. `Xoshiro(seed)`) to give two policies
identical shock draws for a like-for-like comparison.
"""
function simulate(policy, p::RBCParams; T::Int=200, k0=nothing, A0=nothing,
                  rng::AbstractRNG=Random.default_rng())
    ss = steady_state(p)
    kt = something(k0, ss.k)
    At = something(A0, ss.A)

    k = Vector{Float64}(undef, T)
    A = Vector{Float64}(undef, T)
    c = Vector{Float64}(undef, T)
    y = Vector{Float64}(undef, T)
    inv = Vector{Float64}(undef, T)
    shocks = randn(rng, T)

    for t in 1:T
        k[t] = kt
        A[t] = At
        yt = production(kt, At, p.alpha)
        res = yt + (1.0 - p.delta) * kt
        ct = consumption_share(policy, kt, At) * res
        y[t] = yt
        c[t] = ct
        inv[t] = yt - ct
        kt = res - ct
        At = next_productivity(At, shocks[t], p.rho, p.sigma_eps)
    end

    return (capital=k, productivity=A, consumption=c, output=y, investment=inv)
end

"Simulate a [`TIPolicy`](@ref) at its own calibration."
simulate(pol::TIPolicy; kwargs...) = simulate(pol, pol.p; kwargs...)
