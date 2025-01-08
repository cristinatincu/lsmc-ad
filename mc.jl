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

function binomial_tree_am(S0::Float64, K::Float64, r::Float64, σ::Float64, T::Float64, N::Int, option_type::String)
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
    # Time step
    Δt = T / N

    # Up and down factors
    u = exp(σ * sqrt(Δt))
    d = 1 / u

    # Risk-neutral probability
    q = (exp(r * Δt) - d) / (u - d)

    # Discount factor
    disc = exp(-r * Δt)

    # Initialize asset prices at maturity
    S = [S0 * u^(N - i) * d^i for i in 0:N]

    # Initialize option values at maturity
    if option_type == "call"
        option_values = [max(S[i] - K, 0.0) for i in 1:(N + 1)]
    elseif option_type == "put"
        option_values = [max(K - S[i], 0.0) for i in 1:(N + 1)]
    else
        error("Invalid option type. Use 'call' or 'put'.")
    end

    # Backward induction through the tree
    for t in N-1:-1:0
        for i in 1:(t + 1)
            # Calculate continuation value
            continuation_value = disc * (q * option_values[i] + (1 - q) * option_values[i + 1])
            
            # Asset price at node (S0*u^(t-i)*d^i)
            stock_price = S0 * u^(t - i + 1) * d^(i - 1)

            # Exercise value
            exercise_value = option_type == "call" ? max(stock_price - K, 0.0) : max(K - stock_price, 0.0)

            # Take the maximum of continuation and exercise
            option_values[i] = max(continuation_value, exercise_value)
        end
    end

    return option_values[1]  # Option price at the root of the tree
end

# Function to compute paths and calculate the price of an American option using LSMC
function lsmc_am(S0, σ, K, r, T, N, P, option_type::String, plot_regr::Bool)
    # simulate stock price paths
    S = simulate_paths(S0, r, σ, T, N, P)
    return lsmc_am(S, K, r, T, N, P, option_type, plot_regr)
end

# Function to calculate the price of an American option using LSMC
function lsmc_am(S, K, r, T, N, P, option_type::String, plot_regr::Bool)
    df = exp(-r * T / N )  # discount factor per time step

    # Calculate the intrinsic values matrix,
    # needed for determining in-the-money options
    V = option_type == "put" ? max.(K .- S, 0) : max.(S .- K, 0)

    # initialise vector that keeps track of the values at stopping points,
    # starting with discounted option values at maturity
    realised_cash_flow = V[end, :] .* df

    for t in N:-1:2  # iterate backwards through time points
        in_the_money = V[t, :] .!= 0  # consider only in-the-money options
        
        # Construct polynomial basis matrix up to 3rd degree
        A = [x^d for d in 0:3, x in S[t, :]] # each row corresponds to a path

        # Perform least-squares regression to estimate continuation values
        # see https://en.wikipedia.org/wiki/QR_decomposition - julia specific implementation
        β = A[:, in_the_money]' \ realised_cash_flow[in_the_money]

        if plot_regr
            plot_conditional_expectation(S[t, in_the_money], realised_cash_flow[in_the_money], β)
        end 

        # multiply transposed A with β coefficients
        continuation_values = A' * β

        for p in 1:P  # iterate through paths
            exercise_value = max(K - S[t, p], 0)  # exercise value for put option

            # Update option values based on exercise and continuation values
            if in_the_money[p] && continuation_values[p] < exercise_value
                realised_cash_flow[p] = exercise_value * df  # discount the exercise value
            else
                realised_cash_flow[p] *= df  # discount the intrinsic value
            end
        end
    end

    # Return the final option value
    if option_type == "put"
        price = max(mean(realised_cash_flow), K - S[1, 1])
    else
        price = max(mean(realised_cash_flow), S[1, 1] - K)
    end
    print_option_price(price)
    return price
end

# Function to compute delta using finite difference
function lsmc_delta(S0, K, T, r, σ, N, P, option_type::String, h)
    """
    Compute the delta of an American option using finite differences.
    S0: initial asset price
    K: strike price
    T: time to maturity
    r: risk-free rate
    σ: volatility
    N: number of steps
    P: number of paths
    is_put: boolean indicating if the option is a put
    h: small perturbation in asset price
    """
    Random.seed!(42) # minimise noise (because 42 is the answer)

    # Price with S0 + h
    price_up = lsmc_am(S0 + h, σ, K, r, T, N, P, option_type, false)
    # Price with S0 - h
    price_down = lsmc_am(S0 - h, σ, K, r, T, N, P, option_type, false)

    # Central difference approximation for delta
    delta = (price_up - price_down) / (2 * h)
    return delta
end

# Function to compute vega using finite difference
function lsmc_vega(S0, K, T, r, σ, N, P, option_type::String, h)
    """
    Compute the vega of an American option using finite differences.
    S0: initial asset price
    K: strike price
    T: time to maturity
    r: risk-free rate
    σ: volatility
    N: number of steps
    P: number of paths
    is_put: boolean indicating if the option is a put
    h: small perturbation in volatility
    """
    Random.seed!(42) # minimise noise (because 42 is the answer)
    
    # Price with sigma + h
    price_up = lsmc_am(S0, σ + h, K, r, T, N, P, option_type, false)

    # Price with sigma - h
    price_down = lsmc_am(S0, σ - h, K, r, T, N, P, option_type, false)

    # Central difference approximation for vega
    vega = (price_up - price_down) / (2 * h)
    return vega
end