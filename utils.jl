using Plots
using Statistics
import ReverseDiff: Dual


function print_option_price(V::Float64)
    println("LSMC without Reverse Diff option price ", V)
end


function print_option_price(V::Dual{Nothing, Float64, 2})
    println("LSMC with Reverse Diff option price ", V.value, " delta ", V.partials[1], " vega ", V.partials[2])
end


function get_dual_vector_values(v::Vector{Dual{Nothing, Float64, 2}}, return_type::String)::Vector{Float64}
    # Define a dictionary to map return types to their corresponding extraction functions
    extractors = Dict(
        "value" => x -> x.value,
        "delta" => x -> x.partials[1],
        "vega"  => x -> x.partials[2]
    )

    # Check if the return type is valid and extract the corresponding values
    if haskey(extractors, return_type)
        return [extractors[return_type](x) for x in v]
    else
        error("Return type not recognized. It should be 'value', 'delta', or 'vega'.")
    end
end


# plot the conditional expectation of the option value as a function of the stock price
function plot_conditional_expectation(stock_price::Vector{Float64}, realised_cash_flow, β)
    # Define the range of x-values for plotting
    plot_x = range(minimum(stock_price), stop=maximum(stock_price), step=0.01)
    g(x) = sum(β[i] * x^(i-1) for i in 1:4)  # regression function
    display(
        scatter(
            stock_price, # stock price
            realised_cash_flow, # option value
            label="Realised Cash Flow", 
            color="blue",
            xlabel="Stock Price",
            ylabel="Option Value",
            title="Expected cashflow as a Function of Stock Price"
        )
    )
    display(plot!(plot_x, g.(plot_x), label="Regression", color="red", linewidth=5))
end


function plot_conditional_expectation(stock_price::Vector{Dual{Nothing, Float64, 2}}, realised_cash_flow, β)
    # Extract the values from dual numbers
    stock_price_values = get_dual_vector_values(stock_price, "value")
    realised_cash_flow_values = get_dual_vector_values(realised_cash_flow, "value")
    β_values = get_dual_vector_values(β, "value")

    # Define the range of x-values for plotting
    plot_x = range(minimum(stock_price_values), stop=maximum(stock_price_values), step=0.01)
    g(x) = sum(β_values[i] * x^(i-1) for i in 1:length(β))  # regression function

    display(
        scatter(
            stock_price_values, # stock price
            realised_cash_flow_values, # option value
            label="Realised Cash Flow", 
            color="blue",
            xlabel="Stock Price",
            ylabel="Option Value",
            title="Expected cashflow as a Function of Stock Price"
        )
    )
    display(plot!(plot_x, g.(plot_x), label="Regression", color="red", linewidth=5))
end


# Function to calculate the mean and standard error of n simulations
function run_simulations(f, args...; n::Int)
    # Determine the type of the result
    sample_result = f(args...)
    result_type = eltype([sample_result])

    results = Vector{result_type}(undef, n)
    times = Float64[]

    for i in 1:n
        elapsed_time = @elapsed result = f(args...)
        results[i] = result
        push!(times, elapsed_time)
    end

    println("Mean for simulations: ", mean(results))
    println("Standard error for simulations: ", std(results) / sqrt(n))

    println("Mean simulation time: ", mean(times))
    return results, times
end