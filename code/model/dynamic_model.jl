# Solution to the H1B visa misallocation model based on Hopenhayn (1992)

using LinearAlgebra, Statistics, Parameters, Distributions, Optim
using Distributions, Expectations, Interpolations, SparseArrays


##########################################################################
# Model parameters
@with_kw struct Primitives
    β::Float64 = 0.96           # Discount factor
    φ::Float64 = 0.5            # returns to scale
    θ::Float64 = 0.3            # CES parameter
    ζ::Float64 = 1.0            # Foreign labor productivity scale
    ψ::Float64 = 0.5            # Domestic labor supply elasticity
    ψ̄::Float64 = 1.0            # Domestic labor supply scale
    η::Float64 = 2.0            # Lottery variance shifter

    κo::Float64 = 0.5          # Fixed cost of operation
    κe::Float64 = 1.0          # Fixed cost of entry
    κf::Float64 = 0.15          # Fixed cost of hiring foreign labor
    δ::Float64 = 0.16          # Visa expiration rate
    F̄::Float64 = 0.25           # Visa hiring cap
end

# Simulation parameters
@with_kw struct Simulations
    z_grid::Vector{Float64} = [0.25, 1.0] # Productivity states
    n_z::Int = length(z_grid)
    P_z::Matrix{Float64} = [0.9 0.1; 
                            0.1 0.9]  # Transition matrix for z
    ν_z::Vector{Float64} = [0.5, 0.5]        # Initial distribution for z


    f_max::Float64 = 2.0 # foreign labor grids
    n_f::Int = 50
    f_grid::Vector{Float64} = collect(range(0.0, stop=f_max, length=n_f))
    n_nodes::Int = 15 # number of nodes for integration
    d_high::Float64 = 100.0 # domestic labor high guess
    a_tol::Float64 = 1e-4 # minimum application tolerance

    max_iter::Int = 10000 # VFI and simulation max iterations
    tol::Float64 = 1e-8

    ge_max_iter::Int = 50 # GE max iterations
    ge_tol::Float64 = 1e-5
    ge_damping::Float64 = 0.382 # GE damping factor
    W_lower::Float64 = 0.75 # wage search lower bound
    W_upper::Float64 = 2.0 # wage search upper bound
    M_lower::Float64 = 0.01 # entry mass search lower bound
    M_upper::Float64 = 100.0 # entry mass search upper bound
end

# model results storage
@with_kw mutable struct Results 
    # firm problem variables
    d_policy::Matrix{Float64} # static domestic labor policy
    profit::Matrix{Float64} # static profit function
    x_policy::Matrix{Bool} # exit policy
    V::Matrix{Float64} # value function
    V_stay::Matrix{Float64} # value of staying function
    f_policy::Matrix{Float64} # foreign labor vacancy policy
    a_policy::Matrix{Float64} # foreign labor application policy
    g::Matrix{Float64} # distribution over (f,z)
    T_star::SparseMatrixCSC{Float64, Int} # transition matrix for distribution

    E::Matrix{Union{Nothing, Expectation}} # expectation operator

    # general eqbm
    M::Float64 # mass of entrants
    N::Float64 # total mass of firms
    D::Float64 # total domestic labor stock
    F::Float64 # total foreign labor stock
    A::Float64 # total applications for foreign labor
    F₋::Float64 # foreign labor firings
    μ::Float64 # lottery mean success rate
    W::Float64 # wage rate
end


function initialize(; β=0.96, φ=0.5, θ=0.3, ζ=1.0, ψ=0.5, ψ̄=1.0, η=1.5,
                       κo=0.5, κe=1.0, κf=0.12, δ=0.16, F̄=0.25,
                       z_grid=[0.25, 1.0], P_z=[0.9 0.1; 0.1 0.9], ν_z=[0.5, 0.5],
                       f_max=2.0, n_f=50, n_nodes=15,
                       d_high=100.0, a_tol=1e-4,
                       max_iter=10000, tol=1e-7, ge_max_iter=50, ge_tol=1e-5, ge_damping=0.618,
                       W_lower=0.75, W_upper=2.0, M_lower=0.01, M_upper=100.0,
                       μ_init=0.5)
    # initialize the model primitives, simulations, and results
    prim = Primitives(β=β, φ=φ, θ=θ, ζ=ζ, ψ=ψ, ψ̄=ψ̄, η=η,
                      κo=κo, κe=κe, κf=κf, δ=δ, F̄=F̄)
    sim = Simulations(z_grid=z_grid, P_z=P_z, ν_z=ν_z,
                      f_max=f_max, n_f=n_f, n_nodes=n_nodes,
                      d_high=d_high, a_tol=a_tol,
                      max_iter=max_iter, tol=tol,
                      ge_max_iter=ge_max_iter, ge_tol=ge_tol, ge_damping=ge_damping,
                      W_lower=W_lower, W_upper=W_upper, M_lower=M_lower, M_upper=M_upper)
    res = Results(zeros(sim.n_f, sim.n_z), zeros(sim.n_f, sim.n_z), falses(sim.n_f, sim.n_z),
                   zeros(sim.n_f, sim.n_z), zeros(sim.n_f, sim.n_z), zeros(sim.n_f, sim.n_z),
                   zeros(sim.n_f, sim.n_z), ones(sim.n_f, sim.n_z) / (sim.n_f * sim.n_z), 
                   spzeros(Float64, sim.n_f * sim.n_z, sim.n_f * sim.n_z),
                   fill(nothing, sim.n_f, sim.n_f), 
                   1.0, 0.5, 1.0, 1.0, 1.0, 1.0, μ_init, 1.5)
    return prim, sim, res
end


##########################################################################
# Firm's problem functions

function labor_index(d::Float64, f::Float64; prim::Primitives)
    return (d^prim.θ + prim.ζ^(1-prim.θ) * f^prim.θ) ^ (1/prim.θ)
end

function compute_profit(z::Float64, d::Float64, f::Float64; W::Float64, prim::Primitives)
    l = labor_index(d, f; prim=prim)
    revenue = z * l^prim.φ
    cost = W * (d + f) + prim.κo
    #cost = W * l + prim.κo
    return revenue - cost
end

function solve_static_profit(W::Float64, prim::Primitives, sim::Simulations)
    profit = zeros(sim.n_f, sim.n_z)
    d_policy = zeros(sim.n_f, sim.n_z)

    # Create an iterator of all index pairs (i_f, i_z)
    indices = CartesianIndices((1:sim.n_f, 1:sim.n_z))

    #Threads.@threads 
    for idx in indices
        i_f, i_z = idx[1], idx[2]
        
        # Access shared read-only data
        z = sim.z_grid[i_z]
        f = sim.f_grid[i_f]
        opt = optimize(d -> -compute_profit(z, d, f; W=W, prim=prim), 1e-8, sim.d_high)
        if Optim.converged(opt) == false
            error("Optimization did not converge for z=$z, f=$f")
        end
        d_policy[i_f, i_z] = opt.minimizer
        profit[i_f, i_z] = -opt.minimum
    end
    return profit, d_policy
end


function build_expectation(prim::Primitives, sim::Simulations, res::Results)
    @unpack_Primitives prim
    @unpack_Simulations sim
    @unpack_Results res

    cache = Matrix{Union{Nothing, Expectation}}(nothing, sim.n_f, sim.n_f)

    # Thread over both dimensions of the transition matrix (f_current, f_next)
    indices = CartesianIndices((1:sim.n_f, 1:sim.n_f))

    Threads.@threads for idx in indices
        i_f, i_next = idx[1], idx[2]
        f = sim.f_grid[i_f]
        f_next = sim.f_grid[i_next]
        f_decay = (1 - δ) * f
        # Calculate applications a implied by transition
        if f_next > f_decay + a_tol
            a = f_next - f_decay
            α_a = η * μ * a
            β_a = η * (1 - μ) * a
            
            # Create distribution and generate quadrature nodes/weights
            dist = Beta(α_a, β_a) 
            cache[i_f, i_next] = expectation(dist; n=n_nodes) 
        end
    end
    return cache
end


function firm_bellman(prim::Primitives, sim::Simulations, res::Results)
    @unpack_Primitives prim 
    @unpack_Simulations sim
    @unpack_Results res

    f̃_next = zeros(n_f, n_z)
    continuation = fill(-Inf, n_f, n_z)

    V_interps = [LinearInterpolation(f_grid, V[:, i_z]) for i_z in 1:n_z]

    Threads.@threads for i_f in 1:n_f
        f = f_grid[i_f]
        f_decay = (1-δ) * f
        for (i_z, z) in enumerate(z_grid)
            for (i_f_next, f̃) in enumerate(f_grid)
                if f̃ <= f_decay + a_tol
                    candidate = P_z[i_z, :] ⋅ res.V[i_f_next, :]
                    if candidate > continuation[i_f, i_z]
                        continuation[i_f, i_z] = candidate
                        f̃_next[i_f, i_z] = f̃
                    end
                else
                    a = f̃ - f_decay
                    E_op = E[i_f, i_f_next]::Expectation

                    EV = 0.0
                    # calculate expected value over z_next and lottery process
                    for (i_z_next, z_next) in enumerate(z_grid)
                        beta_integral = 0.0
                        @inbounds for k in 1:length(E_op.nodes)
                            p = E_op.nodes[k]
                            weight = E_op.weights[k]
                            beta_integral += weight * V_interps[i_z_next](f_decay + p*a)
                        end
                        #function integrand(p)
                        #    V_interps[i_z_next](f_decay + p*a) 
                        #end
                        EV += P_z[i_z, i_z_next] * beta_integral
                    end

                    candidate  = EV - κf * a * μ
                    if candidate > continuation[i_f, i_z]
                        continuation[i_f, i_z] = candidate
                        f̃_next[i_f, i_z] = f̃
                    end
                end
            end
        end
    end

    x_next = continuation .< 0.0
    V_next = profit .+ β * (1 .- x_next) .* continuation
    V_stay_next = profit .+ β * continuation
    a_policy = (f̃_next .- (1-prim.δ) .* reshape(sim.f_grid, sim.n_f, 1)) .* (1 .- x_next)
    a_policy = max.(a_policy, 0.0)

    return V_next, x_next, f̃_next, V_stay_next, a_policy
end


function VFI!(prim::Primitives, sim::Simulations, res::Results)

    res.profit, res.d_policy = solve_static_profit(res.W, prim, sim)
    res.E = build_expectation(prim, sim, res)

    for iter in 1:sim.max_iter
        V_next, x_next, f̃_next, V_stay_next, a_policy = firm_bellman(prim, sim, res)

        diff = maximum(abs.(V_next .- res.V))
        res.V .= V_next
        res.V_stay .= V_stay_next
        res.x_policy .= x_next
        res.f_policy .= f̃_next
        res.a_policy .= a_policy

        if diff < sim.tol
            println("VFI converged in $iter iterations with diff = $diff")
            return
        end
    end
    error("VFI did not converge within the maximum number of iterations")
end


function build_T_star(prim::Primitives, sim::Simulations, res::Results)
    @unpack_Primitives prim
    @unpack_Simulations sim
    @unpack_Results res

    rows = Int[]
    cols = Int[]
    transition_probs = Float64[]

    f_midpoints = [(sim.f_grid[i] + sim.f_grid[i+1])/2.0 for i in 1:(sim.n_f-1)]
    total_states = sim.n_f * sim.n_z

    for i_z in 1:sim.n_z
        for i_f in 1:sim.n_f
            # If the firm chooses to exit, it does not transition to any state tomorrow.
            # Its row in the transition matrix remains empty (all zeros).
            if res.x_policy[i_f, i_z]
                continue
            end

            source_idx = i_f + (i_z - 1) * sim.n_f
            
            f_curr = sim.f_grid[i_f]
            f_decay = (1.0 - prim.δ) * f_curr
            f_target = res.f_policy[i_f, i_z]
            
            # Case A: Deterministic
            if f_target <= f_decay + sim.a_tol
                idx_f_next = searchsortedfirst(sim.f_grid, f_target - 1e-10)
                
                for i_z_next in 1:sim.n_z
                    prob_z = sim.P_z[i_z, i_z_next]
                    if prob_z > 0.0
                        dest_idx = idx_f_next + (i_z_next - 1) * sim.n_f
                        push!(rows, source_idx)
                        push!(cols, dest_idx)
                        push!(transition_probs, prob_z)
                    end
                end

            # Case B: Lottery
            else
                a = f_target - f_decay
                α_dist = prim.η * res.μ * a
                β_dist = prim.η * (1.0 - res.μ) * a
                dist = Beta(α_dist, β_dist)
                
                for j_f in 1:sim.n_f
                    bin_lower_f = (j_f == 1) ? -Inf : f_midpoints[j_f - 1]
                    bin_upper_f = (j_f == sim.n_f) ? Inf : f_midpoints[j_f]
                    
                    p_lower = clamp((bin_lower_f - f_decay) / a, 0.0, 1.0)
                    p_upper = clamp((bin_upper_f - f_decay) / a, 0.0, 1.0)
                    
                    mass_f = 0.0
                    if p_upper > p_lower
                        mass_f = cdf(dist, p_upper) - cdf(dist, p_lower)
                    end
                    
                    if mass_f > 1e-12
                        for i_z_next in 1:sim.n_z
                            prob_z = sim.P_z[i_z, i_z_next]
                            total_prob = mass_f * prob_z
                            
                            if total_prob > 0.0
                                dest_idx = j_f + (i_z_next - 1) * sim.n_f
                                push!(rows, source_idx)
                                push!(cols, dest_idx)
                                push!(transition_probs, total_prob)
                            end
                        end
                    end
                end
            end
        end
    end

    return sparse(rows, cols, transition_probs, total_states, total_states)
end


function solve_distribution!(prim::Primitives, sim::Simulations, res::Results)
    res.T_star = build_T_star(prim, sim, res)

    # Firms enter with f=0 (index 1) and z drawn from ν_z
    entry_vec = zeros(sim.n_f * sim.n_z)
    
    # Assuming f_grid[1] == 0.0 is the entry level for foreign labor
    i_f_entry = 1 
    
    for i_z in 1:sim.n_z
        # Calculate linear index for (f=0, z=z_i)
        idx = i_f_entry + (i_z - 1) * sim.n_f
        # Assign entry probability mass
        entry_vec[idx] = sim.ν_z[i_z]
    end

    # Iterate until convergence
    # g_next = T(survivors) * g + M * entrants
    for iter in 1:sim.max_iter
        g_next_vec = res.T_star' * reshape(res.g, sim.n_f * sim.n_z) + res.M * entry_vec
        g_next = reshape(g_next_vec, sim.n_f, sim.n_z)

        diff = maximum(abs.(g_next .- res.g))
        res.g .= g_next

        if diff < sim.tol
            res.N = sum(res.g)
            println("Distribution converged in $iter iterations. Diff = $diff, Mass = $(res.M)")
            return
        end
    end
    println("Distribution did not converge within the maximum number of iterations")
end


###########################################################################
# Solve general equilibrium

function GE_residual(prim::Primitives, sim::Simulations, res::Results)
    @unpack_Simulations sim
    @unpack_Results res

    # Total domestic labor demand
    D = res.d_policy ⋅ res.g
    D_residual = D - prim.ψ̄ * res.W ^ (-prim.ψ)

    # Total applications
    A = res.a_policy ⋅ res.g
    μ_residual = μ - prim.F̄ / A

    # free entry condition
    expected_value_entry = sum(res.V[1,:] ⋅ sim.ν_z) - prim.κe

    return [D_residual; μ_residual; expected_value_entry]
end


function solve_GE!(prim::Primitives, sim::Simulations, res::Results)
    
    # Initialize bisection bounds for μ ∈ [0, 1]
    μ_lower = 0.0
    μ_upper = 1.0
    
    for iter_ge in 1:sim.ge_max_iter
        # Bisection step: try midpoint
        res.μ = (μ_lower + μ_upper) / 2.0
        println("\nGE iteration $iter_ge: μ = $(round(res.μ, digits=4)), bounds = [$(round(μ_lower, digits=4)), $(round(μ_upper, digits=4))]")
        
        # Step 1: Given μ, find W satisfying free entry condition E_ν[V(z,0)] = κ_e
        function entry_residual(log_W)
            res.W = exp(log_W)
            VFI!(prim, sim, res)
            expected_value = res.V[1, :] ⋅ sim.ν_z
            return (expected_value - prim.κe)^2
        end
        
        opt_W = optimize(entry_residual, log(sim.W_lower), log(sim.W_upper))
        if Optim.converged(opt_W) == false
            error("Wage optimization did not converge in GE iteration $iter_ge")
        end
        res.W = exp(opt_W.minimizer)
        VFI!(prim, sim, res)
        
        # Step 2: Given W and μ, find M clearing labor market
        function labor_residual(log_M)
            res.M = exp(log_M)
            solve_distribution!(prim, sim, res)
            D_demand = res.d_policy ⋅ res.g
            D_supply = prim.ψ̄ * res.W^prim.ψ
            return (D_demand - D_supply)^2
        end
        
        opt_M = optimize(labor_residual, log(sim.M_lower), log(sim.M_upper))
        if Optim.converged(opt_M) == false
            error("Labor market optimization did not converge in GE iteration $iter_ge")
        end
        res.M = exp(opt_M.minimizer)
        solve_distribution!(prim, sim, res)
        
        # Step 3: Compute aggregates and implied μ
        res.A = res.a_policy ⋅ res.g
        res.D = res.d_policy ⋅ res.g
        res.F = sim.f_grid ⋅ sum(res.g, dims=2)
        
        μ_implied = min(prim.F̄ / res.A, 1.0)
        μ_residual = μ_implied - res.μ
        μ_error = abs(μ_residual)
        
        println("  W=$(round(res.W,digits=3)), M=$(round(res.M,digits=3)), " *
                "A=$(round(res.A,digits=2)), μ_impl=$(round(μ_implied,digits=4)), " *
                "residual=$(round(μ_residual,digits=6))")
        
        # Check convergence
        if μ_error < sim.ge_tol
            println("✓ GE converged in $iter_ge iterations")
            return
        end
        
        # Step 4: Update bisection bounds
        # If μ_implied > μ, the cap is too loose → increase μ (move lower bound up)
        # If μ_implied < μ, the cap is too tight → decrease μ (move upper bound down)
        if μ_residual > 0
            μ_lower = res.μ
        else
            μ_upper = res.μ
        end
        
        # Check if bounds have converged
        if (μ_upper - μ_lower) < sim.ge_tol
            println("✓ GE converged (bisection bounds collapsed) in $iter_ge iterations")
            return
        end
    end
    
    error("GE did not converge within $(sim.ge_max_iter) iterations")
end


##########################################################################

#=
prim, sim, res = initialize(; φ=0.5, θ=0.25, κo=0.5, κf=0.2, F̄=0.2,
                            n_f=51, f_max = 5.0, ge_tol=1e-5, n_nodes=10, μ_init=0.5, max_iter=20000,
                            ge_max_iter=100, W_lower=0.5, W_upper=2.0, M_lower=0.1, M_upper=10.0)
=#

#=
prim, sim, res = initialize(; φ=0.6, θ=0.4, κo=0.5, κf=0.4, F̄=0.15,
                            n_f=50, f_max = 5.0, ge_tol=1e-5, n_nodes=10, μ_init=0.5, max_iter=20000,
                            ge_max_iter=100, W_lower=0.4, W_upper=1.5, M_lower=0.01, M_upper=10.0)

# inner loop
@time VFI!(prim, sim, res)
@time solve_distribution!(prim, sim, res)

# outer loop, general equilibrium
@time solve_GE!(prim, sim, res)




using Plots
plot(sim.f_grid, res.profit, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Profit")

plot(sim.f_grid, res.d_policy, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Domestic Labor d")

plot(sim.f_grid, res.V, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Value Function V")

plot(sim.f_grid, res.f_policy, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Next Period Target f̃", legend=:topright)
plot!(sim.f_grid, (1-prim.δ) * sim.f_grid, l=:dash, label="f' = (1-δ) * f")

plot(sim.f_grid, res.a_policy, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Applications a")

plot(sim.f_grid, res.x_policy, label=["low z" "high z"], xlabel="Foreign Labor f", ylabel="Exit Policy x")

histogram(sim.f_grid, weights=res.g[:,1], bins=30, xlabel="Foreign Labor f", ylabel="Mass of Firms")
histogram!(sim.f_grid, weights=res.g[:,2], bins=30, xlabel="Foreign Labor f", ylabel="Mass of Firms")
=#
