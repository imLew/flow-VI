using DrWatson

if haskey(ENV, "JULIA_ENVIRONMENT")  # on the cluster
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else  # local
    @quickactivate
end

using BSON
# using Distributions
# using DataFrames
using LinearAlgebra
using KernelFunctions
using Plots
# using Distributions
using ValueHistories
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples
const LinReg = Examples.LinearRegression

include("run_funcs.jl")

DIRNAME = "linear_regression"

N_RUNS = 1

problem_params = Dict(
    :n_samples =>[ 20 ],
    :sample_range =>[ [-3, 3] ],
    :true_ϕ =>[ x -> [1, x, x^2] ],
    :true_w =>[ [2, -1, 0.2] ],
    :true_β =>[ 2 ],
    :ϕ =>[ x -> [1, x, x^2] ],
    :μ₀ =>[ zeros(3) ],
    :Σ₀ =>[ 0.1I(3) ],
    :MAP_start =>[ true ],
)

alg_params = Dict(
    :n_iter => [ 500 ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :step_size => [ 0.001 ],
    :n_particles => [ 50 ],
    :update_method => [:forward_euler],
    :α => @onlyif(:update_method == :naive_WAG, [3.1, 3.5, 5] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, .5,] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [3., 1] ),
    :kernel_cb => [ median_trick_cb! ],
    # :callback => [plot_cb]
)

function plot_results(plt, q, problem_params)
    x = range(problem_params[:sample_range]..., length=100)
    for w in eachcol(q)
        model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        plot!(plt,x, LinReg.y(model), alpha=0.3, color=:orange, legend=:none)
    end
    plot!(plt,x, 
            LinReg.y(LinReg.RegressionModel(problem_params[:ϕ], 
                              mean(q, dims=2), 
                              problem_params[:true_β])), 
        color=:red)
    plot!(plt,x, 
            LinReg.y(LinReg.RegressionModel(problem_params[:true_ϕ], 
                              problem_params[:true_w], 
                              problem_params[:true_β])), 
        color=:green)
    return plt
end

ap = dict_list(alg_params)[1]
pp = dict_list(problem_params)[1]
data = run_linear_regression(pp, ap, N_RUNS)

initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
H₀ = Distributions.entropy(initial_dist)
EV = ( 
      LinReg.true_gauss_expectation(initial_dist,  
            LinReg.RegressionModel(data[:ϕ], mean(initial_dist), 
                                   data[:true_β]),
            data[:sample_data])
       + expectation_V(initial_dist, initial_dist) 
       + 0.5 * logdet(2π * data[:Σ₀])
      )
est_logZ_rkhs = estimate_logZ.(H₀, EV, KL_integral(data[:svgd_results][1][1]))

norm_plot = plot(data[:svgd_results][1][1][:ϕ_norm], title = "φ norm", yaxis = :log)
int_plot = plot( estimate_logZ.(H₀, EV, KL_integral(data[:svgd_results][1][1])) ,title = "log Z", label = "",)
hline!(int_plot, 
       [LinReg.regression_logZ(data[:Σ₀], data[:true_β], data[:true_ϕ], data[:sample_data].x)],
      legend=false)

fit_plot = plot_results(plot(size=(300,250)), data[:svgd_results][1][2], data)
plot(fit_plot, norm_plot, int_plot, layout=@layout [f; n i])

# runs = []
# recent_runs = []
# n_sets = dict_list_count(alg_params)*dict_list_count(problem_params)
# for (i, ap) ∈ enumerate(dict_list(alg_params))
#     for (j, pp) ∈ enumerate(dict_list(problem_params))
#         # @show ap[:update_method]
#         # if haskey(ap, :c₁) 
#         #     @show (ap[:c₁], ap[:c₂]) 
#         # end
#         # if haskey(ap, :α) 
#         #     @show ap[:α] 
#         # end
#         println("$(((i-1)*dict_list_count(problem_params)) + j) out of $n_sets")
#         # name = run_gauss_to_gauss(problem_params=pp, alg_params=ap, 
#         #                           n_runs=N_RUNS, DIRNAME=DIRNAME)
#         push!(recent_runs, name)
#         display(plot_convergence(name))
#     end
#     if readline() == "q"
#         break
#     end
# end
# push!(runs, recent_runs)
