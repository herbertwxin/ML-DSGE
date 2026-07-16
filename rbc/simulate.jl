import Pkg
rbc_dir = abspath(@__DIR__)
Pkg.activate(rbc_dir)

using CairoMakie
using Random: Xoshiro, randn

# Inclusion is required so BSON can deserialize the checkpoint's FullRBC types.
isdefined(Main, :FullRBC) || Base.include(Main, joinpath(rbc_dir, "src", "FullRBC.jl"))
import Main.FullRBC: load_checkpoint, TISolver, solve, consumption_share

Base.@kwdef struct RBCParams
    alpha::Float64 = 0.30
    beta::Float64 = 0.99
    delta::Float64 = 0.10
    gamma::Float64 = 2.0
    rho::Float64 = 0.95
    sigma_eps::Float64 = 0.05
end

compare_with_ti = true  # Set false to skip solving time iteration.

solver = load_checkpoint(joinpath(rbc_dir, "rbc_nn.bson"))
model = solver.model  # `load_checkpoint` loads onto the CPU by default.
training_bounds = (
    k_bounds = solver.p.k_bounds,
    A_sigma_mult = solver.p.A_sigma_mult,
    alpha_bounds = solver.p.alpha_bounds,
    beta_bounds = solver.p.beta_bounds,
    delta_bounds = solver.p.delta_bounds,
    rho_bounds = solver.p.rho_bounds,
    gamma_bounds = solver.p.gamma_bounds,
    sigma_eps_bounds = solver.p.sigma_eps_bounds,
)

steady_state_capital(p::RBCParams) =
    ((1 / p.beta - (1 - p.delta)) / p.alpha)^(1 / (p.alpha - 1))
normalize01(x, (low, high)) = (x - low) / (high - low)

function state_support(p::RBCParams, bounds)
    k_ss = steady_state_capital(p)
    k_support = bounds.k_bounds .* k_ss
    sigma_stat = p.sigma_eps / sqrt(max(1e-4, 1 - p.rho^2))
    width = bounds.A_sigma_mult * sigma_stat
    A_support = (exp(-width), exp(width))
    return (; k_support, A_support)
end

function network_input(k, A, p::RBCParams, bounds)
    support = state_support(p, bounds)
    values = (
        normalize01(k, support.k_support),
        normalize01(A, support.A_support),
        normalize01(p.alpha, bounds.alpha_bounds),
        normalize01(p.beta, bounds.beta_bounds),
        normalize01(p.delta, bounds.delta_bounds),
        normalize01(p.rho, bounds.rho_bounds),
        normalize01(p.gamma, bounds.gamma_bounds),
        normalize01(p.sigma_eps, bounds.sigma_eps_bounds),
    )
    return reshape(Float32[values...], 8, 1)
end

function nn_consumption_share(model, k, A, p::RBCParams, bounds)
    return model(network_input(k, A, p, bounds))[1]
end

next_productivity(A, shock, p::RBCParams) =
    exp(p.rho * log(A) + p.sigma_eps * shock)

function simulate_local(share, p::RBCParams;
                        T::Int=200, k0=nothing, A0=nothing, rng=Xoshiro())
    kt = something(k0, steady_state_capital(p))
    At = something(A0, 1.0)

    capital = Vector{Float64}(undef, T)
    productivity = Vector{Float64}(undef, T)
    consumption = Vector{Float64}(undef, T)
    output = Vector{Float64}(undef, T)
    investment = Vector{Float64}(undef, T)
    shocks = randn(rng, T)

    for t in 1:T
        capital[t] = kt
        productivity[t] = At
        yt = At * kt^p.alpha
        resources = yt + (1 - p.delta) * kt
        ct = share(kt, At) * resources

        output[t] = yt
        consumption[t] = ct
        investment[t] = yt - ct
        kt = resources - ct
        At = next_productivity(At, shocks[t], p)
    end

    return (; capital, productivity, consumption, output, investment)
end

p = RBCParams()
nn_share = (k, A) -> nn_consumption_share(model, k, A, p, training_bounds)
nn_paths = simulate_local(nn_share, p; rng=Xoshiro(42))

ti_paths = nothing
if compare_with_ti
    ti_p = Main.FullRBC.RBCParams(
        alpha=p.alpha, beta=p.beta, delta=p.delta,
        gamma=p.gamma, rho=p.rho, sigma_eps=p.sigma_eps,
    )
    ti_policy = solve(TISolver(ti_p))
    ti_share = (k, A) -> consumption_share(ti_policy, k, A)
    ti_paths = simulate_local(ti_share, p; rng=Xoshiro(42))
end

series = (
    ("Capital", :capital, "k"),
    ("Productivity", :productivity, "A"),
    ("Output", :output, "y"),
    ("Consumption", :consumption, "c"),
)
figure = Figure(size=(760, 620))
for (i, (title, field, ylabel)) in enumerate(series)
    row, column = divrem(i - 1, 2) .+ 1
    axis = Axis(figure[row, column]; title, xlabel="Period", ylabel)
    lines!(axis, getproperty(nn_paths, field); label="NN", linewidth=1.5)
    if compare_with_ti
        lines!(axis, getproperty(ti_paths, field); label="TI", linewidth=1.5, linestyle=:dash)
    end
    i == 1 && axislegend(axis; position=:rt, orientation=:horizontal, framevisible=false)
end
investment_axis = Axis(
    figure[3, 1:2]; title="Investment", xlabel="Period", ylabel="i"
)
lines!(investment_axis, nn_paths.investment; label="NN", linewidth=1.5)
if compare_with_ti
    lines!(investment_axis, ti_paths.investment; label="TI", linewidth=1.5, linestyle=:dash)
end
figure
