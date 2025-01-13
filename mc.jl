include("utils.jl")
using Random
using Plots
import ReverseDiff: Dual
using Polynomials
using Statistics
using Distributions

# Define the Black-Scholes formula for an European option
function black_scholes(S0, K, r, T, σ, is_put::Bool)
    d1 = (log(S0 / K) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    if is_put
        return K * exp(-r * T) * cdf(Normal(0, 1), -d2) - S0 * cdf(Normal(0, 1), -d1)
    else
        return S0 * cdf(Normal(0, 1), d1) - K * exp(-r * T) * cdf(Normal(0, 1), d2)
    end
end

# Define the Delta for an European option
function bs_delta(S0, K, r, T, σ, is_put::Bool)
    d1 = (log(S0 / K) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
    if is_put
        return cdf(Normal(0, 1), d1) - 1
    else
        return cdf(Normal(0, 1), d1)        
    end
end

# Define the Vega for a European option
function bs_vega(S0, K, r, T, σ)
    d1 = (log(S0 / K) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
    return S0 * sqrt(T) * pdf(Normal(0, 1), d1)
end


# generate matrix of prices using GBM
function simulate_paths(s0, r, σ, T::Float64, M, P::Int64)
    # calculate in advance the constants
    Δt = T / M # time step
    a = (r - 0.5 * σ^2) * Δt
    b = σ * sqrt(Δt)
    s0_type = typeof(s0) # type of s0, can be float or dual

    paths = zeros(s0_type, M + 1, P) # path matrix
    rands = zeros(Float64, M, 1) # vector of random numbers per path

    # iterate through paths
    # compiler optimisations make non-vectorised code more efficient
    for i in 1:P
        randn!(rands) # generate random numbers
        paths[1, i] = st = s0 # time point 1, i-th path
        for j in 1:M # iterate through time points
            st *= exp(a + b * rands[j])
            paths[j+1, i] = st # price at timepoint j+1, path i
        end
    end

    return paths
end

# Function to calculate the price of an European option using Monte Carlo
function mc_europ(s0, K, r, σ, T, M, P, is_put::Bool)
    paths = simulate_paths(s0, r, σ, T, M, P)
    s0_type = typeof(s0) # type of s0 which can be float or dual
    payoffs = zeros(s0_type, P)
    for i in 1:P
        S_T = paths[end, i] # final price at maturity
        payoffs[i] = is_put ? max(K - S_T, 0) : max(S_T - K, 0)
    end
    
    # Discount the payoffs to present value
    discounted_payoffs = payoffs .* exp(-r * T)
    
    # Calculate the average of the discounted payoffs
    price = mean(discounted_payoffs)
    print_option_price(price)
    return price
end

function binomial_tree(S0::Float64, K::Float64, r::Float64, σ::Float64, T::Float64, N::Int, option_type::String)
    """
    Binomial tree for American option pricing.
    
    Args:
        S0: Initial stock price.
        K: Strike price.
        r: Risk-free interest rate.
        σ: Volatility of the underlying asset.
        T: Time to maturity.
        N: Number of time steps in the binomial tree.
        option_type: "call" or "put" to specify the option type.
        
    Returns:
        Option price.
    """
    dt = T / N
    u = exp(σ * sqrt(dt))      # Up factor
    d = 1 / u                      # Down factor
    p = (exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    discount = exp(-r * dt)        # Discount factor

    # Initialize stock price tree
    stock_tree = zeros(Float64, N + 1, N + 1)
    stock_tree[1, 1] = S0  # Root node

    for t in 1:N
        for i in 1:t
            stock_tree[i + 1, t + 1] = stock_tree[i, t] * d  # Down move
            stock_tree[i, t + 1] = stock_tree[i, t] * u      # Up move
        end
    end

    # Initialize option value tree
    option_tree = zeros(Float64, N + 1, N + 1)

    # Terminal payoff
    for i in 1:N + 1
        option_tree[i, N + 1] = option_type == "put" ? max(K - stock_tree[i, N + 1], 0.0) : max(stock_tree[i, N + 1] - K, 0.0)
    end

    # Backward induction and exercise boundary extraction
    exercise_boundary = fill(NaN, N + 1)

    for t in N:-1:1
        for i in 1:t
            # Continuation value
            continuation = discount * (p * option_tree[i, t + 1] + (1 - p) * option_tree[i + 1, t + 1])

            # Immediate payoff
            immediate_payoff = option_type == "put" ? max(K - stock_tree[i, t], 0.0) : max(stock_tree[i, t] - K, 0.0)

            # Option value at this node
            option_tree[i, t] = max(immediate_payoff, continuation)

            # Update exercise boundary
            if isnan(exercise_boundary[t]) && immediate_payoff > continuation
                exercise_boundary[t] = stock_tree[i, t]
            end
        end
    end

    return option_tree[1, 1], exercise_boundary  # Option price at the root of the tree
end


# Function to compute paths and calculate the price of an American option using LSMC
function lsmc(S0::Union{Float64, Dual}, σ::Union{Float64, Dual}, K::Float64, r::Float64, T::Float64, N::Int64, P::Int64, option_type::String, num_basis, plot_regr::Bool)
    # simulate stock price paths
    S = simulate_paths(S0, r, σ, T, N, P)
    return lsmc(S, K, r, T, N, P, option_type, num_basis, plot_regr)
end

# Function to calculate the price of an American option using LSMC
function lsmc(S::Matrix, K::Float64, r::Float64, T::Float64, N::Int64, P::Int64, option_type::String, num_basis::Int64, plot_regr::Bool)
    df = exp(-r * T / N )  # discount factor per time step

    # Calculate the intrinsic values matrix,
    # needed for determining in-the-money options
    V = option_type == "put" ? max.(K .- S, 0) : max.(S .- K, 0)

    # initialise vector that keeps track of the values at stopping points,
    # starting with discounted option values at maturity
    realised_cash_flow = V[end, :] .* df

    exercise_boundary = zeros(typeof(S[1,1]), N)  # Initialize boundary array

    for t in N:-1:2  # iterate backwards through time points
        in_the_money = V[t, :] .!= 0  # consider only in-the-money options
        
        # Construct polynomial basis matrix up to 3rd degree
        A = [x^d for d in 0:num_basis, x in S[t, :]] # each row corresponds to a path

        # Perform least-squares regression to estimate continuation values
        # see https://en.wikipedia.org/wiki/QR_decomposition - julia specific implementation
        β = A[:, in_the_money]' \ realised_cash_flow[in_the_money]

        if plot_regr
            plot_conditional_expectation(S[t, in_the_money], realised_cash_flow[in_the_money], β)
        end 

        # multiply transposed A with β coefficients
        continuation_values = A' * β

        # store exercise decision in a vector
        exercise_decision = []
        for p in 1:P  # iterate through paths
            immediate_payoff = option_type == "put" ? max(K - S[t, p], 0) : max(S[t, p] - K, 0) # exercise value for put option

            # Update option values based on exercise and continuation values
            if in_the_money[p] && continuation_values[p] < immediate_payoff
                realised_cash_flow[p] = immediate_payoff * df  # discount the exercise value
                push!(exercise_decision, S[t, p])
            else
                realised_cash_flow[p] *= df  # discount the intrinsic value
            end
        end

        # Identify the exercise boundary
        exercise_boundary[t] = length(exercise_decision) > 0 ? minimum(exercise_decision) : NaN
    end

    initial_intrinsic_value = option_type == "put" ? max(K - S[1, 1], 0) : max(S[1, 1] - K, 0)

    # Return the final option value
    return max(mean(realised_cash_flow), initial_intrinsic_value), exercise_boundary
end

# Function to compute delta using finite difference
function lsmc_delta(S0, K, T, r, σ, N, P, option_type::String, num_basis::Int64, h)
    """
    Compute the delta of an American option using finite differences.
    S0: initial asset price
    K: strike price
    T: time to maturity
    r: risk-free rate
    σ: volatility
    N: number of steps
    P: number of paths
    option_type: "put" or "call" option type
    h: small perturbation in asset price
    """
    #Random.seed!(42) # minimise noise (because 42 is the answer)

    # Price with S0 + h
    price_up = lsmc(S0 + h, σ, K, r, T, N, P, option_type, num_basis, false)
    # Price with S0 - h
    price_down = lsmc(S0 - h, σ, K, r, T, N, P, option_type, num_basis, false)

    # Central difference approximation for delta
    return (price_up - price_down) / (2 * h)
end

# Function to compute vega using finite difference
function lsmc_vega(S0, K, T, r, σ, N, P, option_type::String, num_basis::Int64, h)
    """
    Compute the vega of an American option using finite differences.
    S0: initial asset price
    K: strike price
    T: time to maturity
    r: risk-free rate
    σ: volatility
    N: number of steps
    P: number of paths
    option_type: "put" or "call" option type
    h: small perturbation in volatility
    """
    #Random.seed!(42) # minimise noise (because 42 is the answer)
    
    # Price with sigma + h
    price_up = lsmc(S0, σ + h, K, r, T, N, P, option_type, num_basis, false)

    # Price with sigma - h
    price_down = lsmc(S0, σ - h, K, r, T, N, P, option_type, num_basis, false)

    # Central difference approximation for vega
    return (price_up - price_down) / (2 * h)
end