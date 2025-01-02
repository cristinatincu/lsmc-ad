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
function simulate_paths(s0, r, σ, T, M, n_paths)
    # calculate in advance the constants
    Δt = T / M # time step
    a = (r - 0.5 * σ^2) * Δt
    b = σ * sqrt(Δt)
    s0_type = typeof(s0) # type of s0, can be float or dual

    paths = zeros(s0_type, M + 1, n_paths) # path matrix
    rands = zeros(Float64, M, 1) # vector of random numbers per path

    # iterate through paths
    # compiler optimisations make non-vectorised code more efficient
    for i in 1:n_paths
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

# Function to price an American option using the Binomial Tree method
function binomial_tree_am(S0, K, r, T, σ, N, is_put::Bool)
    Δt = T / N
    u = exp(σ * sqrt(Δt))
    d = 1 / u
    p = (exp(r * Δt) - d) / (u - d)
    discount = exp(-r * Δt)

    # Initialize asset prices at maturity
    asset_prices = [S0 * u^j * d^(N - j) for j in 0:N]

    # Initialize option values at maturity
    option_values = is_put ? max.(K .- asset_prices, 0) : max.(asset_prices .- K, 0)

    # Step back through the tree
    for i in N-1:-1:0
        asset_prices = [S0 * u^j * d^(i - j) for j in 0:i]
        option_values = [discount * (p * option_values[j+2] + (1 - p) * option_values[j+1]) for j in 0:i]
        if is_put
            option_values = [max(option_values[j+1], K - price) for (j, price) in enumerate(asset_prices)]
        else
            option_values = [max(option_values[j+1], price - K) for (j, price) in enumerate(asset_prices)]
        end
    end

    return option_values[1]
end

function lsmc_am(S0, σ, K, r, T, N, P, is_put::Bool, plot_regr::Bool)
    # simulate stock price paths
    S = simulate_paths(S0, r, σ, T, N, best_paths)

    df = round(exp(-r * T / N ), digits=5)  # discount factor per time step

    # Calculate the intrinsic values matrix,
    # needed for determining in-the-money options
    V = is_put ? max.(K .- S, 0) : max.(S .- K, 0)

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
    if is_put
        price = max(mean(realised_cash_flow), K - S[1, 1])
    else
        price = max(mean(realised_cash_flow), S[1, 1] - K)
    end
    print_option_price(price)
    return price
end

# Function to compute delta using finite difference
function compute_delta(S0, K, T, r, σ, N, P, is_put::Bool, h)
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
    # Price with S0 + h
    price_up = lsmc_am(S0 + h, σ, K, r, T, N, P, is_put, false)
    # Price with S0 - h
    price_down = lsmc_am(S0 - h, σ + h, K, r, T, N, P, is_put, false)

    # Central difference approximation for delta
    delta = (price_up - price_down) / (2 * h)
    return delta
end

# Function to compute vega using finite difference
function compute_vega(S0, K, T, r, σ, N, P, is_put::Bool, h)
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
    # Price with sigma + h
    price_up = lsmc_am(S0, σ + h, K, r, T, N, P, is_put, false)

    # Price with sigma - h
    price_down = lsmc_am(S0, σ - h, K, r, T, N, P, is_put, false)

    # Central difference approximation for vega
    vega = (price_up - price_down) / (2 * h)
    return vega
end