# Neural-network Euler-residual solver.
# Julia port of `learn_rbc.py::RBCNet` / `RBCSolver`.
#
# *** Fix applied relative to the Python version ***
# The Euler-equation residual is now normalized by the target marginal utility
# before being squared into the training loss (see `compute_residuals` below).
# See README.md, section "Fix applied: normalized Euler residual", for the
# diagnosis that motivated this change.

normalize_x(x, lo, hi) = (x .- lo) ./ (hi .- lo)
denormalize_x(x, lo, hi) = x .* (hi .- lo) .+ lo
normalize_scalar(x::Real, bounds::Tuple) = (x - bounds[1]) / (bounds[2] - bounds[1])

"""
    build_policy_net(input_dim, hidden_dims, output_bias)

`Dense(..., elu)` stack + a final `Dense(..., 1, sigmoid)` layer whose bias is
pre-initialized to `output_bias`, so the untrained policy starts near the
steady-state consumption share. Mirrors `RBCNet` in `learn_rbc.py`.
"""
function build_policy_net(input_dim::Int, hidden_dims::Vector{Int}, output_bias::Float64)
    layers = Any[]
    prev = input_dim
    for h in hidden_dims
        push!(layers, Dense(prev, h, elu))
        prev = h
    end
    out = Dense(prev, 1, sigmoid)
    out.bias .= output_bias
    push!(layers, out)
    return Chain(layers...)
end

"""
    NNSolver

Holds the policy network, optimizer state, structural-parameter bounds
(via `p::RBCParams`) and Gauss-Hermite quadrature nodes/weights used to
evaluate the conditional expectation in the Euler equation.
"""
mutable struct NNSolver
    p::RBCParams
    model::Chain
    opt_state::Any
    n_quad::Int
    z_nodes::Vector{Float64}
    z_weights::Vector{Float64}
    k_ss::Float64
    c_ss::Float64
    y_ss::Float64
    A_ss::Float64
end

function NNSolver(p::RBCParams; lr::Float64=5e-4)
    k_ss, c_ss, y_ss, A_ss = steady_state(p)
    @info "Reference steady-state capital (A=1)" k_ss

    res_ss_init = y_ss + (1.0 - p.delta) * k_ss
    frac_ss_init = c_ss / res_ss_init
    init_bias = log(frac_ss_init / (1.0 - frac_ss_init))
    @info "Output bias init" init_bias frac_ss_init

    model = build_policy_net(8, [64, 64, 64, 64], init_bias) |> f64
    opt_state = Flux.setup(Adam(lr), model)

    n_quad = 7
    nodes, weights = gausshermite(n_quad)
    z_nodes = nodes .* sqrt(2)
    z_weights = weights ./ sqrt(pi)

    return NNSolver(p, model, opt_state, n_quad, z_nodes, z_weights, k_ss, c_ss, y_ss, A_ss)
end

"""
    sample_batch(p::RBCParams, rng, batch_size)

Sample a batch of states and structural parameters, uniformly over their
respective (normalized) training boxes. Returns a `NamedTuple` with the
normalized NN inputs (`8 x batch_size` matrix), physical states `(k, A)`,
physical structural params, and per-sample state bounds. Mirrors
`RBCSolver.sample_batch`.
"""
function sample_batch(p::RBCParams, rng::AbstractRNG, batch_size::Int)
    k_norm         = rand(rng, batch_size)
    alpha_norm     = rand(rng, batch_size)
    beta_norm      = rand(rng, batch_size)
    delta_norm     = rand(rng, batch_size)
    rho_norm       = rand(rng, batch_size)
    gamma_norm     = rand(rng, batch_size)
    sigma_eps_norm = rand(rng, batch_size)
    A_norm         = rand(rng, batch_size)

    if p.hard_region_prob > 0.0
        hard_mask = rand(rng, batch_size) .< p.hard_region_prob
        n_hard = count(hard_mask)
        if n_hard > 0
            beta_norm[hard_mask]  .= p.hard_beta_low_norm .+ (1.0 - p.hard_beta_low_norm) .* rand(rng, n_hard)
            delta_norm[hard_mask] .= p.hard_delta_high_norm .* rand(rng, n_hard)
        end
    end

    alpha     = denormalize_x(alpha_norm,     p.alpha_bounds[1],     p.alpha_bounds[2])
    beta      = denormalize_x(beta_norm,      p.beta_bounds[1],      p.beta_bounds[2])
    delta     = denormalize_x(delta_norm,     p.delta_bounds[1],     p.delta_bounds[2])
    rho       = denormalize_x(rho_norm,       p.rho_bounds[1],       p.rho_bounds[2])
    gamma     = denormalize_x(gamma_norm,     p.gamma_bounds[1],     p.gamma_bounds[2])
    sigma_eps = denormalize_x(sigma_eps_norm, p.sigma_eps_bounds[1], p.sigma_eps_bounds[2])

    one_m = max.(1e-4, 1.0 .- rho .^ 2)
    sigma_stat = sigma_eps ./ sqrt.(one_m)
    w = p.A_sigma_mult .* sigma_stat
    A_low = exp.(-w)
    A_high = max.(exp.(w), A_low .+ 1e-6)

    k_ss, _, _ = steady_state_batch(alpha, beta, delta)
    k_low  = p.k_bounds[1] .* k_ss
    k_high = p.k_bounds[2] .* k_ss

    k = denormalize_x(k_norm, k_low, k_high)
    A = denormalize_x(A_norm, A_low, A_high)

    inputs = permutedims(hcat(k_norm, A_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm))

    return (inputs=inputs, k=k, A=A, k_low=k_low, k_high=k_high, A_low=A_low, A_high=A_high,
            alpha=alpha, beta=beta, delta=delta, rho=rho, gamma=gamma, sigma_eps=sigma_eps)
end

"""
    compute_residuals(model, batch, z_nodes, z_weights)

Normalized Euler-equation residual for a batch produced by [`sample_batch`](@ref).

The raw Euler residual is `resid_raw = E[beta * mu' * R'] - mu` (marginal-utility
units, `mu = c^-gamma`). Across the wide training range of `(alpha, beta, delta,
rho, gamma, sigma_eps)` used here, `mu`'s magnitude varies by orders of
magnitude, so an *unnormalized* MSE loss on `resid_raw` is dominated by whatever
sampled draw happens to have the largest `|mu|` — starving harder regions
(e.g. high beta / low delta / high gamma, which is exactly where the Python
version's `diagnose_divergence.py` sweep found its worst cases) of effective
training signal.

Fix: divide by the target marginal utility `mu`, turning the residual into a
dimensionless *relative* Euler error (economically: a fractional deviation of
today's marginal utility from its optimal, consumption-smoothed value), so all
parameter draws contribute comparably to the loss regardless of curvature/scale.
"""
function compute_residuals(model, batch, z_nodes::Vector{Float64}, z_weights::Vector{Float64})
    inputs = batch.inputs
    k, A = batch.k, batch.A
    k_low, k_high = batch.k_low, batch.k_high
    A_low, A_high = batch.A_low, batch.A_high
    alpha, beta, delta, rho, gamma, sigma_eps = batch.alpha, batch.beta, batch.delta, batch.rho, batch.gamma, batch.sigma_eps

    alpha_norm     = inputs[3, :]
    beta_norm      = inputs[4, :]
    delta_norm     = inputs[5, :]
    rho_norm       = inputs[6, :]
    gamma_norm     = inputs[7, :]
    sigma_eps_norm = inputs[8, :]

    frac = vec(model(inputs))

    resources = A .* (k .^ alpha) .+ (1.0 .- delta) .* k
    c = frac .* resources
    k_next = max.(resources .- c, 1e-8)
    c = max.(c, 1e-6)

    mu = c .^ (-gamma)
    k_next_norm = normalize_x(k_next, k_low, k_high)

    expected_rhs = zero(mu)
    for i in eachindex(z_nodes)
        eps = z_nodes[i]
        w = z_weights[i]
        log_A_next = rho .* log.(max.(A, 1e-8)) .+ sigma_eps .* eps
        A_next = exp.(log_A_next)
        A_next_norm = normalize_x(A_next, A_low, A_high)

        inputs_next = permutedims(hcat(k_next_norm, A_next_norm, alpha_norm, beta_norm, delta_norm, rho_norm, gamma_norm, sigma_eps_norm))

        frac_next = vec(model(inputs_next))
        resources_next = A_next .* (k_next .^ alpha) .+ (1.0 .- delta) .* k_next
        k_next_next = max.((1.0 .- frac_next) .* resources_next, 1e-8)
        c_next = max.(resources_next .- k_next_next, 1e-6)
        mu_next = c_next .^ (-gamma)
        R_next = alpha .* A_next .* (k_next .^ (alpha .- 1.0)) .+ (1.0 .- delta)
        expected_rhs = expected_rhs .+ w .* (beta .* mu_next .* R_next)
    end

    resid_raw = expected_rhs .- mu
    resid = resid_raw ./ (mu .+ 1e-8)   # <- normalization fix
    return resid
end

"""
    simulate(solver::NNSolver; T=200, k0=nothing, A0=nothing,
             alpha=nothing, beta=nothing, delta=nothing, rho=nothing, gamma=nothing,
             sigma_eps=nothing, rng=Random.default_rng())

Simulate the economy under the trained NN policy at a given structural
calibration (defaults to `solver.p`). No clamping is applied to `(k, A)`
between periods, matching the Python version. Mirrors `RBCSolver.simulate`.
"""
function simulate(solver::NNSolver; T::Int=200, k0=nothing, A0=nothing,
                   alpha=nothing, beta=nothing, delta=nothing, rho=nothing, gamma=nothing,
                   sigma_eps=nothing, rng::AbstractRNG=Random.default_rng())
    p = solver.p
    alpha = alpha === nothing ? p.alpha : alpha
    beta  = beta  === nothing ? p.beta  : beta
    delta = delta === nothing ? p.delta : delta
    rho   = rho   === nothing ? p.rho   : rho
    gamma = gamma === nothing ? p.gamma : gamma
    sigma_eps = sigma_eps === nothing ? p.sigma_eps : sigma_eps

    k_ss_sim, c_ss_sim, y_ss_sim = steady_state_batch(alpha, beta, delta)
    k0 = k0 === nothing ? k_ss_sim : k0
    A0 = A0 === nothing ? solver.A_ss : A0

    k = zeros(T + 1)
    A = zeros(T + 1)
    c = zeros(T)
    y = zeros(T)
    inv = zeros(T)
    k[1] = k0
    A[1] = A0
    eps = randn(rng, T)

    k_low = p.k_bounds[1] * k_ss_sim
    k_high = p.k_bounds[2] * k_ss_sim
    A_low, A_high = a_support_from_shock_params(rho, sigma_eps, p.A_sigma_mult, solver.A_ss)

    alpha_n     = normalize_scalar(alpha, p.alpha_bounds)
    beta_n      = normalize_scalar(beta,  p.beta_bounds)
    delta_n     = normalize_scalar(delta, p.delta_bounds)
    rho_n       = normalize_scalar(rho,   p.rho_bounds)
    gamma_n     = normalize_scalar(gamma, p.gamma_bounds)
    sigma_eps_n = normalize_scalar(sigma_eps, p.sigma_eps_bounds)

    for t in 1:T
        kt, At = k[t], A[t]
        kn = (kt - k_low) / (k_high - k_low)
        an = (At - A_low) / (A_high - A_low)
        state = reshape([kn, an, alpha_n, beta_n, delta_n, rho_n, gamma_n, sigma_eps_n], 8, 1)
        frac = solver.model(state)[1]

        yt = At * kt^alpha
        y[t] = yt
        resources = yt + (1.0 - delta) * kt
        ct = frac * resources
        c[t] = ct
        inv[t] = yt - ct
        k[t+1] = max((1.0 - delta) * kt + yt - ct, 1e-6)

        logA_next = rho * log(max(At, 1e-8)) + sigma_eps * eps[t]
        A[t+1] = exp(logA_next)
    end

    return (capital=k[1:T], productivity=A[1:T], consumption=c, output=y, investment=inv)
end

"""
    train!(solver::NNSolver; batch_size=2048, epochs=10000, eval_every=200,
           val_batch_size=8192, patience=20, min_rel_improve=5e-3,
           panel_n_cases=4, panel_T=120, panel_seed=321,
           best_checkpoint_path=nothing, rng=Random.default_rng())

Train with early stopping on a fixed validation residual batch, reporting a
fixed-panel NN-vs-TI diagnostic (NRMSE, level ratios) every `eval_every`
epochs. Mirrors `RBCSolver.train`.
"""
function train!(solver::NNSolver;
                 batch_size::Int=2048, epochs::Int=10000, eval_every::Int=200,
                 val_batch_size::Int=8192, patience::Int=20, min_rel_improve::Float64=5e-3,
                 panel_n_cases::Int=4, panel_T::Int=120, panel_seed::Int=321,
                 best_checkpoint_path::Union{Nothing,String}=nothing,
                 rng::AbstractRNG=Random.default_rng())
    losses = Float64[]
    val_batch = sample_batch(solver.p, rng, val_batch_size)
    panel = build_validation_panel(solver.p, panel_n_cases, panel_seed)
    if !isempty(panel)
        @info "Validation panel: $(panel_n_cases) fixed parameter cases (T=$(panel_T))"
    end

    best_val_loss = Inf
    best_epoch = 0
    bad_evals = 0
    best_state = nothing

    @info "Training over wide parameter range (normalized Euler residual loss)..."
    for epoch in 1:epochs
        batch = sample_batch(solver.p, rng, batch_size)
        loss_val, grads = Flux.withgradient(solver.model) do m
            r = compute_residuals(m, batch, solver.z_nodes, solver.z_weights)
            mean(abs2, r)
        end
        Flux.update!(solver.opt_state, solver.model, grads[1])
        push!(losses, loss_val)

        if epoch % eval_every == 0
            val_res = compute_residuals(solver.model, val_batch, solver.z_nodes, solver.z_weights)
            val_loss = mean(abs2, val_res)

            panel_metrics = evaluate_validation_panel(solver, panel, panel_T)
            if panel_metrics !== nothing
                lr = panel_metrics["level_ratio"]
                @info "epoch" epoch train_mse=loss_val val_mse=val_loss mean_nrmse=panel_metrics["aggregate"]["mean_nrmse"] max_nrmse=panel_metrics["aggregate"]["max_nrmse"] level_ratio_c=lr["consumption"] level_ratio_k=lr["capital"] level_ratio_y=lr["output"] level_ratio_i=lr["investment"]
            else
                @info "epoch" epoch train_mse=loss_val val_mse=val_loss
            end

            rel_improve = (best_val_loss - val_loss) / max(abs(best_val_loss), 1e-12)
            if val_loss < best_val_loss && (isinf(best_val_loss) || rel_improve >= min_rel_improve)
                best_val_loss = val_loss
                best_epoch = epoch
                bad_evals = 0
                best_state = deepcopy(Flux.state(solver.model))
                if best_checkpoint_path !== nothing
                    save_checkpoint(solver, best_checkpoint_path)
                end
            else
                bad_evals += 1
                if bad_evals >= patience
                    @info "Early stopping" epoch best_val_loss best_epoch
                    break
                end
            end
        end
    end

    if best_state !== nothing
        Flux.loadmodel!(solver.model, best_state)
        @info "Restored best model" best_epoch best_val_loss
    else
        @warn "No best validation checkpoint captured; keeping last-epoch weights."
    end

    return losses
end

"""
    build_validation_panel(p::RBCParams, n_cases::Int, seed::Int)

Build a fixed validation panel: TI policies are solved once (in parallel over
Julia threads, if available) and reused so diagnostics are deterministic and
cheap to re-evaluate during training. Mirrors `RBCSolver._build_validation_panel`.
"""
function build_validation_panel(p::RBCParams, n_cases::Int, seed::Int)
    n_cases <= 0 && return NamedTuple[]
    rng = Xoshiro(seed)
    params_list = [sample_params_uniform(p, rng) for _ in 1:n_cases]
    panel = Vector{Any}(undef, n_cases)
    Threads.@threads for idx in 1:n_cases
        p_i = params_list[idx]
        ti = TISolver(p_i)
        policy = solve!(ti)
        panel[idx] = (params=p_i, ti=ti, policy=policy, seed=seed + idx)
    end
    return panel
end

"""
    evaluate_validation_panel(solver::NNSolver, panel, T::Int)

Average NN-vs-TI gap metrics across the fixed validation panel. Mirrors
`RBCSolver._evaluate_validation_panel`.
"""
function evaluate_validation_panel(solver::NNSolver, panel, T::Int)
    isempty(panel) && return nothing

    metrics_list = Vector{Any}(undef, length(panel))
    for (i, item) in enumerate(panel)
        nn_res = simulate(solver; T=T, alpha=item.params.alpha, beta=item.params.beta,
                           delta=item.params.delta, rho=item.params.rho, gamma=item.params.gamma,
                           sigma_eps=item.params.sigma_eps, rng=Xoshiro(item.seed))
        ti_res = simulate(item.ti, item.policy; T=T, rng=Xoshiro(item.seed))
        metrics_list[i] = gap_metrics(nn_res, ti_res; series=(:consumption, :capital, :output, :investment))
    end

    series = ("consumption", "capital", "output", "investment")
    avg = Dict{String,Any}()
    lvl = Dict{String,Any}()
    for s in series
        avg[s] = Dict(
            "rmse" => mean(m[s]["rmse"] for m in metrics_list),
            "nrmse" => mean(m[s]["nrmse_vs_ti_std"] for m in metrics_list),
            "level_ratio" => mean(m[s]["level_ratio"] for m in metrics_list),
        )
        lvl[s] = avg[s]["level_ratio"]
    end
    avg["level_ratio"] = lvl
    avg["aggregate"] = Dict(
        "mean_nrmse" => mean(m["aggregate"]["mean_nrmse"] for m in metrics_list),
        "max_nrmse" => maximum(m["aggregate"]["max_nrmse"] for m in metrics_list),
    )
    return avg
end

"""
    save_checkpoint(solver::NNSolver, path)

Save the model weights (`Flux.state`) and `RBCParams` calibration/bounds to a
BSON file so `load_checkpoint` can restore an equivalent `NNSolver` without
retraining. Mirrors `RBCSolver.save`.
"""
function save_checkpoint(solver::NNSolver, path::AbstractString)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    model_state = Flux.state(solver.model)
    p = solver.p
    BSON.@save path model_state p
end

"""
    load_checkpoint(path) -> NNSolver

Inverse of [`save_checkpoint`](@ref). Mirrors `RBCSolver.load`.
"""
function load_checkpoint(path::AbstractString)
    data = BSON.load(path)
    p = data[:p]
    solver = NNSolver(p)
    Flux.loadmodel!(solver.model, data[:model_state])
    @info "Loaded model from $path"
    return solver
end
