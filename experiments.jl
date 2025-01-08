include("mc.jl")
include("utils.jl")
import ReverseDiff: Dual
using Plots, Printf

# European Option using BS
# bs = black_scholes(S0, K, r, T, σ, is_put)
# delta = bs_delta(S0, K, r, T, σ, is_put)
# vega = bs_vega(S0, K, r, T, σ)
# 
# println("BS put option price: ", bs, " delta ", delta, " vega ", vega)

# European Option using Monte Carlo
# time_points = Int(252 * T) # number of time points
# prices = [] # store the option price for each number of paths
# for i in paths
#     price = mc_europ(S0, K, r, σ, T, time_points, i, is_put)
#     push!(prices, price)
# end
# plot!(paths, prices, label="MC")


# European Option using Monte Carlo and AD
# mc_europ_ad = mc_europ(Dual(S0, 1, 0), K, r, Dual(σ, 0, 1), T, time_points, best_paths, is_put)

# # Calculate error for Monte Carlo with AD
# error = abs(mc_europ_ad.value - bs)

##################################################################
# Test implementation of LSMC for American options
##################################################################
S0 = 100.0 # current price, has to be float to let the compiler do optimisations
r = 0.0475 # risk-free rate
σ = 0.2 # volatility
K = 110.0 # strike price
T = 1.0 # time to maturity in years
option_type = "put"
paths = [1000, 5000, 10000, 50000, 100000, 150000, 200000] # number of paths
time_points = 252 # number of time points

# lsmc with floats
S = simulate_paths(S0, r, σ, T, time_points, 100)
plot(S, 
    legend=false, 
    title="Stock Price Paths", 
    xlabel="Time", 
    ylabel="Stock Price")

option_type = "put"

# use the paper example to check the implementation
lsmc_paper_paths = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
                     1.09 1.16 1.22 0.93 1.11 0.76 0.92 0.88;
                     1.08 1.26 1.07 0.97 1.56 0.77 0.84 1.22;
                     1.34 1.54 1.03 0.92 1.52 0.9 1.01 1.34]
# this one should be 0.114 when using second degree polynomial
V_lsmc_am = lsmc_am(lsmc_paper_paths, 1.1, 0.06, 3, 3, 8, option_type, false)

# use the paper example with AD
# convert the floats to dual numbers first
lsmc_paper_paths_dual = [Dual(x, 1.0, 0.0) for x in lsmc_paper_paths]
# the result below has to match the one above
V_lsmc_am_ad = lsmc_am(lsmc_paper_paths_dual, 1.1, 0.06, 3, 3, 8, option_type, false)

# Binomial tree for American option
bt_am = binomial_tree_am(1.0, 1.1, 0.06, 0.25, 3.0, 3, option_type)

##################################################################
# Choose number of paths for LSMC
##################################################################

# compute the American option price using binomial tree
bt_am = binomial_tree_am(S0, K, r, σ, T, time_points, option_type)
plot(paths, 
    fill(bt_am, length(paths)), 
    ylims=(11.0,14.00),
    label="Binomial Tree",
    xlabel="Paths",
    ylabel="Option price",
    title="Binomial Tree vs LSMC")

prices_list = [] # store the mean option price for each number of paths
prices_err = [] # store the standard error for each number of paths
times = Dict() # store the simulation times for each number of paths

num_simulations = 100
# run each number of paths num_simulations times
for i in paths
    println("Number of paths: $i")
    prices, times[i] = run_simulations(lsmc_am, S0, σ, K, r, T, time_points, i, option_type, false, n=num_simulations)

    # Statistical error analysis
    mean_price = mean(prices)
    std_dev = std(prices)
    std_error = std_dev / sqrt(num_simulations)

    # Confidence interval (95%)
    lower_bound = mean_price - 1.96 * std_error
    upper_bound = mean_price + 1.96 * std_error

    # Display results
    println("Mean Option Price: $mean_price")
    println("Standard Deviation: $std_dev")
    println("Standard Error: $std_error")
    println("95% Confidence Interval: ($lower_bound, $upper_bound)")

    push!(prices_list, mean_price)
    push!(prices_err, std_error)
end

plot!(paths, 
    prices_list,
    marker=:o, 
    label="LSMC"
    )

sorted_path_keys = sort(collect(keys(times)))
mean_times = [mean(times[k]) for k in sorted_path_keys]
plot(paths, 
    mean_times, 
    marker=:o, 
    label="Mean Time",
    title="Simulation Times",
    xlabel="Paths",
    ylabel="Time (s)",)

# Histogram of simulation times
plots = []
for k in sorted_path_keys
    push!(plots, 
        histogram(times[k], 
            bins=30,
            title="$k paths",
            xlabel="Time (s)",
            ylabel="Frequency",
            size=(1000, 800),
            )
    )
end

plot(plots..., layout=(3, 3), legend=false)

# Plot standard error as a percentage of the option price
plot(
    paths, 
    prices_err ./ prices_list * 100, 
    marker=:o, 
    label=false,
    title="Standard Error of the Option Price",
    xlabel="Paths",
    ylabel="Error (%)",
)

# Display results
println("Paths\tMean Price\tStandard Error\tTime (s)")
for i in 1:length(paths)
    println(@sprintf("%d\t%.4f\t\t%.2f\t\t%.2f", paths[i], prices_list[i], prices_err[i] / prices_list[i] * 100, mean(times[paths[i]])))
end

# Set the number of paths based on the error analysis and time
best_paths = 50000

##################################################################
# AD vs Finite Differences for Pricing and Greeks
##################################################################

# LSMC for American option
lsmc_am_no_ad, lsmc_am_no_ad_times = run_simulations(lsmc_am, S0, σ, K, r, T, time_points, best_paths, option_type, false, n=num_simulations)

# Calculate delta using finite differences
delta_h = S0 * 1e-3 # 0.1% perturbation of the price
delta_fd = lsmc_delta(S0, K, T, r, σ, time_points, best_paths, option_type, delta_h)

# plot lsmc delta as a function of asset price S0
S0_values = 80:1:200
delta_put_values = [lsmc_delta(S0, K, T, r, σ, time_points, best_paths, "put", delta_h) for S0 in S0_values]
delta_call_values = [lsmc_delta(S0, K, T, r, σ, time_points, best_paths, "call", delta_h) for S0 in S0_values]
plot(S0_values, delta_put_values, label="Delta Put", xlabel="Asset Price", ylabel="Delta")
plot!(S0_values, delta_call_values, label="Delta Call")

# Calculate vega using finite differences
vega_h = 0.01 # σ perturbation
vega_fd = lsmc_vega(S0, K, T, r, σ, time_points, best_paths, option_type, vega_h)

# Plot vega as a function of asset price S0
vega_values = [lsmc_vega(S0, K, T, r, σ, time_points, 50000, option_type, vega_h) for S0 in S0_values]
plot(S0_values, vega_values, label="LSMC Vega", xlabel="Asset Price", ylabel="Vega")

# LSMC with dual numbers from reverse diff package.
# the result includes the option price, delta and vega first-order Greeks
lsmc_am_ad, lsmc_am_ad_times = run_simulations(lsmc_am, Dual(S0, 1, 0), Dual(σ, 0, 1), K, r, T, time_points, best_paths, option_type, false, n=num_simulations)

# Plot the delta and vega as a function of asset price S0
lsmc_ad_call_values = [lsmc_am(Dual(float(S0), 1, 0), Dual(σ, 0, 1), K, r, T, time_points, 50000, "call", false) for S0 in S0_values]
lsmc_ad_put_values = [lsmc_am(Dual(float(S0), 1, 0), Dual(σ, 0, 1), K, r, T, time_points, 50000, "put", false) for S0 in S0_values]

lsmc_ad_delta_put_values = [x.partials[1] for x in lsmc_ad_put_values]
lsmc_ad_delta_call_values = [x.partials[1] for x in lsmc_ad_call_values]
plot(S0_values, lsmc_ad_delta_put_values, label="AD Delta Put", xlabel="Asset Price", ylabel="Delta")
plot!(S0_values, lsmc_ad_delta_call_values, label="AD Delta Call")

lsmc_ad_vega_values = [x.partials[2] for x in lsmc_ad_put_values]
plot(S0_values, lsmc_ad_vega_values, label="AD Vega", xlabel="Asset Price", ylabel="Vega")

# Extract the option price, delta and vega from AD results
price_ad = map(x -> x.value, lsmc_am_ad)
delta_ad = map(x -> x.partials[1], lsmc_am_ad)
vega_ad = map(x -> x.partials[2], lsmc_am_ad)

# Calculate error for LSMC with AD
lsmc_ad_price_err = abs(mean(price_ad) - mean(lsmc_am_no_ad)) / mean(lsmc_am_no_ad) * 100
lsmc_ad_delta_err = abs(mean(delta_ad) - delta_fd) / abs(delta_fd) * 100
lsmc_ad_vega_err = abs(mean(vega_ad) - vega_fd) / vega_fd * 100