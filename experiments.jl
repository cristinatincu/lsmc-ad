include("mc.jl")
include("utils.jl")
import ReverseDiff: Dual

S0 = 100.0 # current price, has to be float to let the compiler do optimisations
r = 0.0475 # risk-free rate
σ = 0.2 # volatility
K = 110.0 # strike price
T = 1.0 # time to maturity in years
is_put = false

# European Option using BS
bs = black_scholes(S0, K, r, T, σ, is_put)
delta = bs_delta(S0, K, r, T, σ, is_put)
vega = bs_vega(S0, K, r, T, σ)
paths = [1000, 5000, 10000, 20000, 30000, 50000, 60000, 80000, 100000] # number of paths in increments of 1000 from 1000 to 20000
plot(paths, 
    fill(bs, length(paths)), 
    ylims=(0.05,10.00),
    label="BS",
    xlabel="Paths",
    ylabel="Option price",
    title="MC vs BS")
println("BS put option price: ", bs, " delta ", delta, " vega ", vega)

# European Option using Monte Carlo
time_points = Int(252 * T) # number of time points
option_price = [] # store the option price for each number of paths
for i in paths
    price = mc_europ(S0, K, r, σ, T, time_points, i, is_put)
    push!(option_price, price)
end
plot!(paths, option_price, label="MC")

# Choose best performing number of paths
best_paths = paths[argmin(abs.(option_price .- bs))]

# European Option using Monte Carlo and AD
mc_europ_ad = mc_europ(Dual(S0, 1, 0), K, r, Dual(σ, 0, 1), T, time_points, best_paths, is_put)

# Calculate error for Monte Carlo with AD
error = abs(mc_europ_ad.value - bs)

# use the paper example to match results
# lsmc_paper_paths = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
#                     1.09 1.16 1.22 0.93 1.11 0.76 0.92 0.88;
#                     1.08 1.26 1.07 0.97 1.56 0.77 0.84 1.22;
#                     1.34 1.54 1.03 0.92 1.52 0.9 1.01 1.34]
# # this one should be 0.114 when using second degree polynomial
# V = lsmc_am(lsmc_paper_paths, 1.1, 0.6, 3, 3, 8, true, true)

# # use the paper example with AD
# # convert the floats to dual numbers first
# lsmc_paper_paths_dual = [Dual(x, 1.0, 0.0) for x in lsmc_paper_paths]
# V = lsmc_am(lsmc_paper_paths_dual, 1.1, 0.6, 3, 3, 8, true, true)


# lsmc with floats
plot_regr = false
S = simulate_paths(S0, r, σ, T, time_points, best_paths)
plot(S[:, 1:100], 
    legend=false, 
    title="Stock Price Paths", 
    xlabel="Time", 
    ylabel="Stock Price")

# Binomial tree for American option
bt_am = run_simulations(binomial_tree_am, S0, K, r, T, σ, time_points, is_put, n=10)

# LSMC for American option
lsmc_am_no_ad = run_simulations(lsmc_am, S0, σ, K, r, T, time_points, best_paths, is_put, plot_regr, n=10)

# Calculate delta using finite differences
h = 1.0
delta_fd = run_simulations(compute_delta, S0, K, T, r, σ, time_points, best_paths, is_put, h, n=100)

# Calculate vega using finite differences
vega_fd = run_simulations(compute_vega, S0, K, T, r, σ, time_points, best_paths, is_put, h, n=100)

# lsmc with dual numbers from reverse diff.
# the result includes the option price, delta and vega first-order Greeks
lsmc_am_ad = run_simulations(lsmc_am, Dual(S0, 1, 0), Dual(σ, 0, 1), K, r, T, time_points, best_paths, is_put, false, n=10)
