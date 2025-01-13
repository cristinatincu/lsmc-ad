include("mc.jl")
include("utils.jl")
import ReverseDiff: Dual
using Plots, Printf
using JLD2

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

# use the paper example to check the implementation
lsmc_paper_paths = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
                     1.09 1.16 1.22 0.93 1.11 0.76 0.92 0.88;
                     1.08 1.26 1.07 0.97 1.56 0.77 0.84 1.22;
                     1.34 1.54 1.03 0.92 1.52 0.9 1.01 1.34]
# this one should be 0.114 when using second degree polynomial
V_lsmc = lsmc(lsmc_paper_paths, 1.1, 0.06, 3, 3, 8, option_type, 3, false)

# use the paper example with AD
# convert the floats to dual numbers first
lsmc_paper_paths_dual = [Dual(x, 1.0, 0.0) for x in lsmc_paper_paths]
# the result below has to match the one above
V_lsmc_ad = lsmc(lsmc_paper_paths_dual, 1.1, 0.06, 3, 3, 8, option_type, 3, false)

# Binomial tree for American option
bt, _ = binomial_tree(1.0, 1.1, 0.06, 0.25, 3.0, 3, option_type)

##################################################################
# Choose number of paths for LSMC
##################################################################

# compute the American option price using binomial tree
bt_am, bt_boundary = binomial_tree(S0, K, r, σ, T, time_points, option_type)
plot(paths, 
    fill(bt_am, length(paths)), 
    ylims=(11.0, 13.0),
    label="Binomial Tree",
    xlabel="Paths",
    ylabel="Option price",
    title="Binomial Tree vs LSMC")

prices_list = [] # store the mean option price for each number of paths
prices_rmse = [] # store the standard error for each number of paths
prices_error = [] # store the standard error for each number of paths
times = Dict() # store the simulation times for each number of paths

num_simulations = 100
# run each number of paths num_simulations times
for i in paths
    prices, times[i] = run_simulations(lsmc, S0, σ, K, r, T, time_points, i, option_type, 3, false, n=num_simulations)

    # Statistical error analysis
    println("Statistical analysis for number of paths: $i")
    statistical_analysis(prices, num_simulations, bt_am)
    rmse = sqrt(mean((prices .- bt_am).^2))
    mean_price = mean(prices)
    std_dev = std(prices)
    std_error = std_dev / sqrt(num_simulations)

    # Confidence interval (95%)
    lower_bound = mean_price - 1.96 * std_error
    upper_bound = mean_price + 1.96 * std_error


    push!(prices_list, mean_price)
    push!(prices_rmse, rmse)
    push!(prices_error, std_error)
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

plot(plots..., layout=(4, 2), legend=false)

# Plot standard error as a percentage of the option price
plot(
    paths, 
    prices_rmse, 
    marker=:o, 
    label=false,
    title="RMSE of the Option Price",
    xlabel="Paths",
    ylabel="Error (%)",
)

# Plot standard error vs times
plot(
    mean_times, 
    prices_rmse, 
    marker=:o, 
    label=false,
    title="RMSE vs Time",
    xlabel="Time (s)",
    ylabel="Error (%)",
)

# Plot standard error as a percentage of the option price
plot(
    paths, 
    prices_error ./ prices_list * 100, 
    marker=:o, 
    label=false,
    title="Standard Error of the Option Price",
    xlabel="Paths",
    ylabel="Error (%)",
)

# Plot standard error vs times
plot(
    mean_times, 
    prices_error ./ prices_list * 100, 
    marker=:o, 
    label=false,
    title="Standard Error vs Time",
    xlabel="Time (s)",
    ylabel="Error (%)",
)

# Display results
println("Paths\tMean Price\tRMSE\tTime (s)")
for i in 1:length(paths)
    println(@sprintf("%d\t%.4f\t\t%.2f\t\t%.2f", paths[i], prices_list[i], prices_rmse[i], mean(times[paths[i]])))
end

# Set the number of paths based on the error analysis and time
best_paths = 50000

# plot pricing bias for each number of paths
plot(paths, 
    abs.(prices_list .- bt_am) ./ bt_am * 100, 
    marker=:o, 
    label=false,
    title="Pricing Bias",
    xlabel="Paths",
    ylabel="Error (%)",
)

# Display results
println("Paths\tPricing Bias")
for i in 1:length(paths)
    println(@sprintf("%d\t%.2f", paths[i], abs(prices_list[i] - bt_am) / bt_am * 100))
end

# save results to disk
save("paths.jld2", "paths", paths, "prices_list", prices_list, "prices_rmse", prices_rmse, "prices_error", prices_error, "times", times,
    "bt_am", bt_am, "best_paths", best_paths)

##################################################################
# Choose polynomial function for LSMC
##################################################################

# compute the American option price using binomial tree
bt, bt_boundary = binomial_tree(S0, K, r, σ, T, time_points, option_type)

# compute the LSMC option price and exercise_boundary
v_lsmc_2, lsmc_boundary_2 = lsmc(S0, σ, K, r, T, time_points, best_paths, option_type, 2, false)
v_lsmc_3, lsmc_boundary_3 = lsmc(S0, σ, K, r, T, time_points, best_paths, option_type, 3, false)

# Compare Boundaries
differences = abs.(bt_boundary .- lsmc_boundary_2)
filtered_differences = filter(x -> !isnan(x) && x != 0.0, differences)

mae = mean(filtered_differences)
max_ae = maximum(filtered_differences)

println("Mean Absolute Error (MAE): $mae")
println("Max Absolute Error (MaxAE): $max_ae")

# Plot Comparison
plot(1:time_points, bt_boundary, label="Binomial Boundary", legend=false)
plot(1:time_points, lsmc_boundary_2, label="LSMC Boundary 2")
plot!(1:time_points, lsmc_boundary_3, label="LSMC Boundary 3", linestyle=:dash)
xlabel!("Time Step")
ylabel!("Stock Price")
title!("Exercise Boundary Comparison")

##################################################################
# AD vs Finite Differences for Pricing and Greeks
##################################################################

# LSMC for American option
alsmc_no_ad, lsmc_no_ad_times = run_simulations(lsmc, S0, σ, K, r, T, time_points, best_paths, option_type, 3, false, n=num_simulations)

# Calculate delta using finite differences
delta_h = 1 # 1% perturbation of the price
delta_fd, delta_fd_times = run_simulations(lsmc_delta, S0, K, T, r, σ, time_points, best_paths, option_type, 3, delta_h, n=num_simulations)
println("Statistical analysis for delta using finite differences")
statistical_analysis(delta_fd, num_simulations)

# Calculate vega using finite differences
vega_h = 0.01 # σ perturbation
vega_fd, vega_fd_times = run_simulations(lsmc_vega, S0, K, T, r, σ, time_points, best_paths, option_type, 3, vega_h, n=num_simulations)
println("Statistical analysis for vega using finite differences")
statistical_analysis(vega_fd, num_simulations)

# LSMC with dual numbers from reverse diff package.
# the result includes the option price, delta and vega first-order Greeks
lsmc_ad, lsmc_ad_times = run_simulations(lsmc, Dual(S0, 1, 0), Dual(σ, 0, 1), K, r, T, time_points, best_paths, option_type, 3, false, n=num_simulations)
# Extract the option price, delta and vega from AD results
price_ad = [x.value for x in lsmc_ad]
delta_ad = map(x -> x.partials[1], lsmc_ad)
vega_ad = map(x -> x.partials[2], lsmc_ad)
println("Statistical analysis for LSMC with AD pricing")
statistical_analysis(price_ad, num_simulations, bt)
println("Statistical analysis for delta using AD")
statistical_analysis(delta_ad, num_simulations, delta_fd)
println("Statistical analysis for vega using AD")
statistical_analysis(vega_ad, num_simulations, vega_fd)

# plot lsmc delta as a function of asset price S0
S0_values = 50:1:170
delta_put_values = [lsmc_delta(float(S0), K, T, r, σ, time_points, best_paths, "put", 3, delta_h) for S0 in S0_values]
delta_call_values = [lsmc_delta(float(S0), K, T, r, σ, time_points, best_paths, "call", 3, delta_h) for S0 in S0_values]
plot(S0_values, delta_call_values, label="FD Delta Call", xlabel="Asset Price", ylabel="Delta")
plot!(S0_values, delta_put_values, label="FD Delta Put")

# Plot vega as a function of asset price S0
vega_values = [lsmc_vega(float(S0), K, T, r, σ, time_points, best_paths, option_type, 3, vega_h) for S0 in S0_values]
plot(S0_values, vega_values, label="FD Vega", xlabel="Asset Price", ylabel="Vega")

# Plot the delta and vega as a function of asset price S0
lsmc_ad_call_values = [lsmc(Dual(float(S0), 1, 0), Dual(σ, 0, 1), K, r, T, time_points, best_paths, "call", 3, false) for S0 in S0_values]
lsmc_ad_put_values = [lsmc(Dual(float(S0), 1, 0), Dual(σ, 0, 1), K, r, T, time_points, best_paths, "put", 3, false) for S0 in S0_values]

lsmc_ad_delta_put_values = [x.partials[1] for x in lsmc_ad_put_values]
lsmc_ad_delta_call_values = [x.partials[1] for x in lsmc_ad_call_values]
plot(S0_values, lsmc_ad_delta_put_values, label="AD Delta Put", xlabel="Asset Price", ylabel="Delta")
plot!(S0_values, lsmc_ad_delta_call_values, label="AD Delta Call")

lsmc_ad_vega_values = [x.partials[2] for x in lsmc_ad_put_values]
plot(S0_values, lsmc_ad_vega_values, label="AD Vega", xlabel="Asset Price", ylabel="Vega")

# save the results to disk
save("results.jld2", "lsmc_no_ad", lsmc_no_ad, "lsmc_no_ad_times", lsmc_no_ad_times,
    "delta_fd", delta_fd, "delta_fd_times", delta_fd_times,
    "vega_fd", vega_fd, "vega_fd_times", vega_fd_times,
    "lsmc_ad", lsmc_ad, "lsmc_ad_times", lsmc_ad_times,
    "price_ad", price_ad, "delta_ad", delta_ad, "vega_ad", vega_ad,
    "bt", bt, "bt_boundary", bt_boundary, "lsmc_boundary_2", lsmc_boundary_2,
    "lsmc_boundary_3", lsmc_boundary_3, "best_paths", best_paths,
    "delta_put_values", delta_put_values, "delta_call_values", delta_call_values,
    "vega_values", vega_values, "lsmc_ad_delta_put_values", lsmc_ad_delta_put_values,
    "lsmc_ad_delta_call_values", lsmc_ad_delta_call_values, "lsmc_ad_vega_values", lsmc_ad_vega_values)

# load data from disk
data = load("results.jld2")