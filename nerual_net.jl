### Julia environment setup
using Flux, Statistics, Random, Parameters, Plots, DataFrames
using BasisMatrices, QuantEcon, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)
### Model parameters for the RBC model (all Float32 for GPU compatibility)
@with_kw mutable struct Params
    α::Float32 = 0.30      # capital share in production
    β::Float32 = 0.95      # discount factor
    δ::Float32 = 0.1       # depreciation rate
    γ::Float32 = 2.0       # CRRA utility coefficient (risk aversion)
    ρ::Float32 = 0.90      # persistence of productivity shock
    σ_ε::Float32 = 0.02      # std dev of shock innovation (for log A)


    #bounds
    k_bounds::Tuple{Float32,Float32} = (0.5f0, 1.5f0) #fraction of SS capital
    A_bounds::Tuple{Float32,Float32} = (0.5f0, 1.5f0)
    β_bounds::Tuple{Float32,Float32} = (0.9f0, 0.99f0)
    ρ_bounds::Tuple{Float32,Float32} = (0.9f0, 0.99f0)
    γ_bounds::Tuple{Float32,Float32} = (0.5f0, 4.0f0)
end

# marginal utility function for CRRA utility
function u_prime(c, γ)
    return c .^ (-γ)      # derivative of c^(1-γ)/(1-γ) is c^(-γ)
end

# Steady-state (for A=1) for reference (solve α β (A) k^(α-1) + β(1-δ) = 1)
function steady_state(α, β, δ)
    A_ss = 1.0f0
    k_ss = @. ((1.0f0 / β - (1.0f0 - δ)) / (α * A_ss))^(1.0f0 / (α - 1.0f0))
    y_ss = @. A_ss * k_ss^α
    c_ss = @. y_ss - δ * k_ss

    return k_ss, c_ss, y_ss, A_ss
end

para = Params()
k_ss, c_ss, y_ss, A_ss = steady_state(para.α, para.β, para.δ)
println("Steady-state capital (A=1) ≈ ", round.(k_ss, digits=3))

# Pre-compute Quadrature nodes (Constant)
# We use 5 nodes for expectation
const z_nodes_global, z_weights_global = qnwnorm(7, 0.0, 1.0)


#Normalize variables
function normalize(x, x_low, x_high)
    return (x .- x_low) ./ (x_high - x_low)
end
function denormalize(x, x_low, x_high)
    return x .* (x_high - x_low) .+ x_low
end

# Calculate steady-state fraction for initialization
k_ss_init, c_ss_init, y_ss_init, A_ss_init = steady_state(para.α, para.β, para.δ)
res_ss_init = y_ss_init + (1.0f0 - para.δ) * k_ss_init
frac_ss_init = c_ss_init / res_ss_init
init_bias = -log(1.0f0 / frac_ss_init - 1.0f0)
println("Initializing output bias to: $init_bias (SS frac: $frac_ss_init)")

#Now construct the neural network
model = Chain(
    Dense(5, 128, elu),
    Dense(128, 64, elu),
    Dense(64, 32, elu),
    Dense(32, 1, sigmoid; bias=[init_bias])  # output layer
)

"""
    sample_batch(para::Params, batch_size)

Generates sample of state variables
"""
function sample_batch(para::Params, batch_size)
    @unpack σ_ε, ρ, k_bounds, A_bounds, β_bounds = para
    # We sample uniform in log(k) to ensure lower end has coverage (since distribution of k might be skewed)
    k_batch = rand(Float32, batch_size)
    β_batch = rand(Float32, batch_size)
    ρ_batch = rand(Float32, batch_size)
    γ_batch = rand(Float32, batch_size)
    # sample productivity A (log-normal around 1)
    # We approximate the ergodic distribution of A_t as lognormal with mean 0 and std derived from σ_ε.
    # To be precise, if A follows log AR(1), its stationary distribution is N(0, σ_stat^2) with σ_stat^2 = σ_ε^2 / (1-ρ^2).
    σ_stat = σ_ε ./ sqrt.(1.0f0 .- ρ .^ 2.0f0)
    A = exp.(σ_stat .* randn(Float32, batch_size))
    A_batch = normalize(A, A_bounds[1], A_bounds[2])

    return Float32.(k_batch), Float32.(A_batch), Float32.(β_batch), Float32.(ρ_batch), Float32.(γ_batch)
end

function euler_residual_batch(para, model, data)
    @unpack α, β, δ, γ, σ_ε, ρ, k_bounds, A_bounds, β_bounds, ρ_bounds, γ_bounds = para
    k_batch, A_batch, β_batch, ρ_batch, γ_batch = data
    β = denormalize(β_batch, β_bounds[1], β_bounds[2])
    ρ = denormalize(ρ_batch, ρ_bounds[1], ρ_bounds[2])
    γ = denormalize(γ_batch, γ_bounds[1], γ_bounds[2])

    # Re-calculate SS inside loop or use passed globals?
    # Calling steady_state is cheap.
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)

    batch_size = length(k_batch)
    k_low = k_bounds[1] * k_ss
    k_high = k_bounds[2] * k_ss

    X = hcat(k_batch, A_batch, β_batch, ρ_batch, γ_batch)' # shape 5 x batch_size
    # model(X) will give a 1 x N matrix (or vector) of outputs
    model_out = model(X)            # 1 x batch_size

    #output fraction of resources consumed
    frac = vec(model_out)  # reshape to 1D vector length = batch_size

    #denormalize variables
    k = denormalize(k_batch, k_low, k_high)
    A = denormalize(A_batch, A_bounds[1], A_bounds[2])
    # Use broadcasting to compute vectorized consumption given states
    c = frac .* (A .* (k .^ α) .+ (1.0f0 .- δ) .* k)

    # Now compute Euler residual for each element
    # u'(c) = c^(-γ)
    #marginal_utility = c .^ (-γ)
    # Next state capital k' for each state:
    k_next = (1.0f0 .- δ) .* k .+ A .* (k .^ α) .- c
    k_next = clamp.(k_next, k_low, k_high) #NECESSARY FOR CONVERGENCE
    c = (1.0f0 .- δ) .* k .+ A .* (k .^ α) .- k_next
    marginal_utility = c .^ (-γ)

    # Expectation calculation using Quadrature (aligning with TI)
    n_quad = 5
    # Use pre-computed globals to avoid Zygote differentiation issues
    z_nodes = z_nodes_global
    z_weights = z_weights_global

    # Use list comprehension + sum to avoid mutation (compatible with Zygote)
    expected_rhs = sum(
        begin
            eps = Float32(z_nodes[i_z])
            weight = Float32(z_weights[i_z])

            # Next state A'
            A_next = exp.(log.(A) .* ρ .+ σ_ε .* eps)
            A_next = clamp.(A_next, A_bounds[1], A_bounds[2])

            # Normalize for network input
            A_next_norm = normalize(A_next, A_bounds[1], A_bounds[2])
            k_next_norm = normalize(k_next, k_low, k_high)

            # Predict policy at (k', A')
            X_next = hcat(k_next_norm, A_next_norm, β_batch, ρ_batch, γ_batch)'
            frac_next = vec(model(X_next))

            # Compute c'
            resource_next = A_next .* (k_next .^ α) .+ (1.0f0 .- δ) .* k_next

            # Compute k'' from policy
            k_next_next = (1.0f0 .- frac_next) .* resource_next
            k_next_next = clamp.(k_next_next, k_low, k_high)

            c_next = resource_next - k_next_next
            c_next = max.(c_next, 1.0f-6) # numerical safety

            # Marginal utility
            marginal_utility_next = c_next .^ (-γ)

            # Gross Return
            R_next = α .* A_next .* (k_next .^ (α .- 1.0f0)) .+ (1.0f0 .- δ)

            # Weighted term for this node
            weight .* (β .* marginal_utility_next .* R_next)
        end
        for i_z in 1:n_quad
    )

    # Residual = LHS - E[RHS]
    residuals = expected_rhs .- marginal_utility
    return residuals  # vector of length batch_size
end

# Define loss function
function loss_fn(model, data)
    residuals = euler_residual_batch(para, model, data)
    return mean(residuals .^ 2.0f0)
end

# Hyperparameters
batch_size = 2048      # number of state points per batch
epochs = 10000         # training epochs
η = 5e-4               # initial learning rate

# Initialize optimizer
opt_state = Flux.setup(Adam(η), model) #ADAM is an efficient optimizer
# Training loop
println("Starting training...")
losses = []
for epoch in 1:epochs
    # Sample a batch of states
    data = sample_batch(para, batch_size)
    # Compute gradient of loss (mean squared residual) w.r.t. model parameters
    val, grads = Flux.withgradient(model) do m
        loss_fn(m, data)
    end
    Flux.update!(opt_state, model, grads[1])
    # (Optional) decay learning rate or print progress
    push!(losses, val)
    if epoch % 100 == 0
        # Compute current loss for reporting
        println("Epoch $epoch, training MSE Euler residual = $(round(val, sigdigits=3))")
    end
end

plot(losses[200:end], title="Training Loss", xlabel="Epochs", ylabel="MSE Euler Residual", legend=false)
savefig("traning_loss.png")


# Function to simulate the economy using the trained model
function simulate_economy(model, para::Params, T::Int=200, k0=nothing, A0=nothing)
    @unpack α, β, δ, γ, ρ, σ_ε, k_bounds, A_bounds, β_bounds, ρ_bounds, γ_bounds = para
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)
    k_low = k_bounds[1] * k_ss
    k_high = k_bounds[2] * k_ss
    # If initial values not provided, use steady state values
    if isnothing(k0)
        k0 = k_ss
    end
    if isnothing(A0)
        A0 = A_ss
    end

    # Initialize arrays to store time series
    k_series = zeros(Float32, T + 1)
    A_series = zeros(Float32, T + 1)
    c_series = zeros(Float32, T)
    y_series = zeros(Float32, T)
    i_series = zeros(Float32, T)

    # Set initial values
    k_series[1] = k0
    A_series[1] = A0

    # Generate all random shocks in advance
    ε_series = randn(Float32, T)

    # Simulation loop
    for t in 1:T
        # Current state
        k = k_series[t]
        A = A_series[t]

        # Normalize state for model input
        k_norm = normalize(k, k_low, k_high)
        A_norm = normalize(A, A_bounds[1], A_bounds[2])
        β_norm = normalize(β, β_bounds[1], β_bounds[2])
        ρ_norm = normalize(ρ, ρ_bounds[1], ρ_bounds[2])
        γ_norm = normalize(γ, γ_bounds[1], γ_bounds[2])

        # Create input for neural network
        state = reshape([k_norm, A_norm, β_norm, ρ_norm, γ_norm], 5, 1)

        # Get consumption fraction from model
        frac = model(state)[1]

        # Calculate current output
        y = A * k^α
        y_series[t] = y

        # Calculate consumption
        resources = y + (1.0f0 - δ) * k
        c = frac * resources
        c_series[t] = c

        # Calculate investment
        i = y - c
        i_series[t] = i

        # Next-period capital - FIX: proper capital accumulation equation
        k_next = (1.0f0 - δ) * k + i  # This was incorrect before (was just i)
        k_series[t+1] = max(k_next, 1e-6)  # Prevent negative capital

        # Next-period productivity (AR(1) process)
        logA_next = ρ * log(A) + σ_ε * ε_series[t]
        A_series[t+1] = exp(logA_next)
    end

    # Create DataFrame with results
    results = DataFrame(
        capital=k_series[1:T],
        productivity=A_series[1:T],
        consumption=c_series,
        output=y_series,
        investment=i_series,
        savings_rate=i_series ./ y_series
    )

    # Add metadata as properties
    results.k_ss = k_ss * ones(T)
    results.c_ss = c_ss * ones(T)
    results.y_ss = y_ss * ones(T)
    results.A_ss = A_ss * ones(T)
    results.β = β * ones(T)
    results.ρ = ρ * ones(T)

    return results
end

# Function to compute impulse response to productivity shock
function impulse_response(model, para::Params, shock_size=0.05, T=40, NIRF=20)
    k_ss, c_ss, y_ss, A_ss = steady_state(para.α, para.β, para.δ)
    # Run two simulations: one baseline and one with a shock
    # Initial conditions: steady state
    Random.seed!(96)
    baseline = simulate_economy(model, para, T, k_ss, A_ss)
    for i in 1:NIRF-1
        baseline = baseline .+ simulate_economy(model, para, T, k_ss, A_ss)
    end
    baseline = baseline ./ NIRF
    # Shocked economy: shock_size% higher productivity initially
    Random.seed!(96)
    shocked = simulate_economy(model, para, T, k_ss, A_ss * (1 + shock_size))
    for i in 1:NIRF-1
        shocked = shocked .+ simulate_economy(model, para, T, k_ss, A_ss * (1 + shock_size))
    end
    shocked = shocked ./ NIRF

    # Calculate percentage deviations from baseline
    deviations = DataFrame()

    # Calculate deviations for each variable
    deviations.output = 100 * (shocked.output ./ baseline.output .- 1)
    deviations.consumption = 100 * (shocked.consumption ./ baseline.consumption .- 1)
    deviations.investment = 100 * (shocked.investment ./ baseline.investment .- 1)
    deviations.capital = 100 * (shocked.capital ./ baseline.capital .- 1)

    # Add metadata
    deviations.β = baseline.β
    deviations.ρ = baseline.ρ

    # Plot impulse responses
    p = plot(title="Impulse Responses to $(shock_size*100)% Productivity Shock (β=$(round(baseline.β[1],digits=3)), ρ=$(round(baseline.ρ[1],digits=3)))",
        xlabel="Time", ylabel="% Deviation from Baseline",
        legend=:topright, size=(800, 500))

    plot!(p, 0:T-1, deviations.output, label="Output", linewidth=2)
    plot!(p, 0:T-1, deviations.consumption, label="Consumption", linewidth=2)
    plot!(p, 0:T-1, deviations.investment, label="Investment", linewidth=2)
    plot!(p, 0:T-1, deviations.capital, label="Capital", linewidth=2)

    return p, deviations
end

# Generate impulse response
println("\nComputing impulse response to productivity shock...")
ir_plot, deviations = impulse_response(model, para, 0.05, 40, 50)
# display(ir_plot)

savefig("NN_Impulse_Response.png")
println("Figure saved.")

# ==========================================
# COMPARISON: Time Iteration vs Neural Network
# ==========================================

"""
    solve_ti(para::Params; tol=1e-6, max_iter=1000)

Solves the RBC model using **Time Iteration** with BasisMatrices.
Matches the method in ramsey_TI.jl but for the NN model.
"""
function solve_ti(para::Params; tol=1e-6, max_iter=1000)
    @unpack α, β, δ, γ, ρ, σ_ε, k_bounds, A_bounds = para

    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)

    # Grids
    # We use the bounds from Params
    k_min = k_bounds[1] * k_ss
    k_max = k_bounds[2] * k_ss
    A_min = A_bounds[1] * A_ss
    A_max = A_bounds[2] * A_ss

    # Create Basis
    n_k = 30
    n_A = 15

    # Spline basis for (k, A)
    basis = Basis(SplineParams(LinRange(k_min, k_max, n_k), 0, 3),
        SplineParams(LinRange(A_min, A_max, n_A), 0, 3))

    # Nodes
    nodes_tuple = nodes(basis)[2]
    k_nodes = nodes_tuple[1]
    A_nodes = nodes_tuple[2]

    # Initial guess: c = c_ss
    c_init = fill(c_ss, length(k_nodes), length(A_nodes))
    policy_c = Interpoland(basis, vec(c_init))

    # Quadrature
    n_quad = 7
    z_nodes, z_weights = qnwnorm(n_quad, 0.0, 1.0)

    c_new_vals = similar(c_init)

    println("Starting Time Iteration (BasisMatrices)...")

    # Damping factor to prevent oscillation
    # 1.0 = full update (unstable), 0.1 = slow but stable
    damping = 0.5

    for iter in 1:max_iter

        # Loop over nodes
        for i in 1:length(k_nodes)
            for j in 1:length(A_nodes)
                k = k_nodes[i]
                A = A_nodes[j]

                # 1. Evaluate current policy
                c_old = policy_c([k, A])
                c_old = max(c_old, 1e-6)

                # Resource constraint
                y = A * k^α
                res = y + (1 - δ) * k

                # Ensure c is feasible for k' calculation
                # If c_old is too high, k' becomes negative.
                # We clamp c_old effectively for the purpose of finding k'
                c_calc = min(c_old, res - 1e-4)

                k_prime = res - c_calc

                # Clamp k_prime to grid bounds to avoid wild extrapolation
                # This is crucial for stability with splines
                k_prime_clamped = clamp(k_prime, k_min, k_max)

                # Expectation
                E_rhs = 0.0
                for z in 1:n_quad
                    eps = z_nodes[z]
                    logA_prime = ρ * log(A) + σ_ε * eps
                    A_prime = exp(logA_prime)

                    # Clamp A_prime as well
                    A_prime_clamped = clamp(A_prime, A_min, A_max)

                    # Interpolate future policy
                    c_prime = policy_c([k_prime_clamped, A_prime_clamped])
                    c_prime = max(c_prime, 1e-6)

                    # Euler RHS
                    mu_prime = c_prime^(-γ)
                    R_prime = α * A_prime * k_prime^(α - 1) + (1 - δ)

                    E_rhs += z_weights[z] * (mu_prime * R_prime)
                end

                # Update c using Euler inverse
                # c_target = (β * E)^(-1/γ)
                c_target = (β * E_rhs)^(-1 / γ)

                # Apply Damping
                c_new_vals[i, j] = damping * c_target + (1 - damping) * c_old
            end
        end

        # Check convergence
        # Evaluate current policy at nodes to compare
        current_vals = [policy_c([k, A]) for k in k_nodes, A in A_nodes]
        diff = maximum(abs.(c_new_vals - current_vals))

        if iter % 10 == 0
            println("TI Iteration $iter, diff: $diff")
        end

        if diff < tol
            println("TI Converged in $iter iterations")
            return policy_c
        end

        # Update policy
        policy_c = Interpoland(basis, vec(c_new_vals))
    end

    println("TI Did not converge")
    return policy_c
end

function simulate_ti(policy_c, para::Params, T::Int=200, k0=nothing, A0=nothing)
    @unpack α, β, δ, γ, ρ, σ_ε = para
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)

    if isnothing(k0)
        k0 = k_ss
    end
    if isnothing(A0)
        A0 = A_ss
    end

    k_series = zeros(Float32, T + 1)
    A_series = zeros(Float32, T + 1)
    c_series = zeros(Float32, T)
    y_series = zeros(Float32, T)
    i_series = zeros(Float32, T)

    k_series[1] = k0
    A_series[1] = A0

    # Removed Random.seed!(123) to allow external control
    ε_series = randn(Float32, T)

    for t in 1:T
        k = k_series[t]
        A = A_series[t]

        c = policy_c([k, A])
        c_series[t] = c

        y = A * k^α
        y_series[t] = y

        i = y - c
        i_series[t] = i

        k_next = y + (1 - δ) * k - c
        k_series[t+1] = k_next

        logA_next = ρ * log(A) + σ_ε * ε_series[t]
        A_series[t+1] = exp(logA_next)
    end

    results = DataFrame(
        capital=k_series[1:T],
        productivity=A_series[1:T],
        consumption=c_series,
        output=y_series,
        investment=i_series
    )

    # Add metadata
    results.β = β * ones(T)
    results.ρ = ρ * ones(T)

    return results
end

# Function to compute impulse response for TI
function impulse_response_ti(policy_c, para::Params, shock_size=0.05, T=40, NIRF=20)
    k_ss, c_ss, y_ss, A_ss = steady_state(para.α, para.β, para.δ)

    # Baseline
    Random.seed!(96)
    baseline = simulate_ti(policy_c, para, T, k_ss, A_ss)
    for i in 1:NIRF-1
        baseline = baseline .+ simulate_ti(policy_c, para, T, k_ss, A_ss)
    end
    baseline = baseline ./ NIRF

    # Shocked
    Random.seed!(96)
    shocked = simulate_ti(policy_c, para, T, k_ss, A_ss * (1 + shock_size))
    for i in 1:NIRF-1
        shocked = shocked .+ simulate_ti(policy_c, para, T, k_ss, A_ss * (1 + shock_size))
    end
    shocked = shocked ./ NIRF

    # Deviations
    deviations = DataFrame()
    deviations.output = 100 * (shocked.output ./ baseline.output .- 1)
    deviations.consumption = 100 * (shocked.consumption ./ baseline.consumption .- 1)
    deviations.investment = 100 * (shocked.investment ./ baseline.investment .- 1)
    deviations.capital = 100 * (shocked.capital ./ baseline.capital .- 1)

    p = plot(title="TI Impulse Responses to $(shock_size*100)% Productivity Shock",
        xlabel="Time", ylabel="% Deviation from Baseline",
        legend=:topright, size=(800, 500))

    plot!(p, 0:T-1, deviations.output, label="Output", linewidth=2)
    plot!(p, 0:T-1, deviations.consumption, label="Consumption", linewidth=2)
    plot!(p, 0:T-1, deviations.investment, label="Investment", linewidth=2)
    plot!(p, 0:T-1, deviations.capital, label="Capital", linewidth=2)

    return p, deviations
end

# Run TI
println("\nRunning Time Iteration...")
policy_ti = solve_ti(para)

# Simulate TI
Random.seed!(123) # Set seed here for the single simulation comparison
ti_results = simulate_ti(policy_ti, para, 200)

# Simulate NN (using existing function)
Random.seed!(123) # EXACTLY SAME SEED AS TI
nn_results = simulate_economy(model, para, 200)

# Plot Comparison
p_comp = plot(title="Comparison: NN vs TI", layout=(2, 1), size=(800, 600))
plot!(p_comp[1], nn_results.consumption, label="NN Consumption", linewidth=2)
plot!(p_comp[1], ti_results.consumption, label="TI Consumption", linewidth=2, linestyle=:dash)
plot!(p_comp[2], nn_results.capital, label="NN Capital", linewidth=2)
plot!(p_comp[2], ti_results.capital, label="TI Capital", linewidth=2, linestyle=:dash)

# display(p_comp)

savefig("TI_vs_NN.png")

# Generate TI Impulse Response
println("\nComputing TI impulse response...")
ir_plot_ti, deviations_ti = impulse_response_ti(policy_ti, para, 0.05, 40, 50)
# display(ir_plot_ti)

savefig("TI_Impulse_Response.png")
