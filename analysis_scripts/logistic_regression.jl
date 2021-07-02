###### Cell ###### - load dependencies
using DrWatson
@quickactivate
using Plots
using ValueHistories
using Distributions
using LinearAlgebra
using BSON
using KernelFunctions
# using PDMats
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind

using Utils
using Examples
# LogReg = LogisticRegression

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

plotdir = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/bayesian_logistic_regression/"

###### Cell ###### - Evaluation of Laplace start
# ALG_PARAMS = Dict(
#     :update_method => [ :forward_euler ],
#     :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
#     :kernel_cb => [ median_trick_cb! ],
#     :step_size => [ 0.01, 0.001 ],
#     :n_iter => [ 1000 ],
#     :n_particles => [ 50 ],
#     :n_runs => [ 10 ],
#     :dKL_estimator => [ :RKHS_norm ],
#     )

# PROBLEM_PARAMS = Dict(
#     :problem_type => [ :logistic_regression ],
#     :MAP_start => [  true, false ],
#     :Laplace_start => [ false,  true ],
#     :n_dim => [ 2 ],
#     :n₀ => [ 50 ],
#     :n₁ => [ 50 ],
#     :μ₀ => [ [0., 0] ],
#     :μ₁ => [ [4., 3] ],
#     :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
#     :Σ₁ => [ [.5 0.1; 0.1 .2] ],
#     :μ_prior => [ zeros(3) ],
#     :Σ_prior => [ I(3) ],
#     :μ_initial => [ [1., 1, 1] ],
#     :Σ_initial => [ I(3) ],
#     :therm_params => [Dict(
#                           :nSamples => 3000,
#                           :nSteps => 30
#                          )],
#     :random_seed => [ 0 ],
# )

###### Cell ###### - load data
all_data = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplacevNormal"))
# for d in all_data
#     d[:Σ_initial] = PDMat(Symmetric(d[:Σ_initial]))
#     d[:Σ_prior] = PDMat(Symmetric(d[:Σ_prior]))
# end

for d in all_data
    display(plot_convergence(d))
    show_params(d)
    readline()
end

# 𝔼[V]=Inf when not using MAP start of Laplace start, so remove thos from analysis
pop!(all_data)
pop!(all_data)
# also MAP_start=true means no Laplace_start, so we can remove the last two which
# are redundant
pop!(all_data)
pop!(all_data)

for d in all_data
    if d[:MAP_start] && d[:Laplace_start]
        d[:Laplace_start] = false
    end
end

###### Cell ###### -
# We are left with 4 experiments, all of them using MAP start and half using
# Laplace covariance, and the step sizes 0.01 and 0.001
for d in all_data
    readline()
    plt = plot_convergence(d)
    show_params(d)
    # Plots.savefig(joinpath(plotdir, "classification_"*get_savename(d))*".png")
    display(plt)
end

for d in all_data
    @show d[:Laplace_start]
    @show mean(d[:estimated_logZ])
    @show std(d[:estimated_logZ])
end

###### Cell ###### -
# We are left with 4 experiments, all of them using MAP start and half using
# Laplace covariance, and the step sizes 0.01 and 0.001
#
data = filter_by_dict(Dict(:step_size => [0.001]), all_data)
for d in data
    plt = plot_convergence(d)
    show_params(d)
    # Plots.savefig(joinpath(plotdir, get_savename(d))*".png")
    # display(plt)
    # readline()
end

# reruns
###### Cell ###### - params
# PROBLEM_PARAMS = Dict(
#     :problem_type => [ :logistic_regression ],
#     :MAP_start => [ true ],
#     :Laplace_start => [ false,  true ],
#     :n_dim => [ 2 ],
#     :n₀ => [ 50 ],
#     :n₁ => [ 50 ],
#     :μ₀ => [ [0., 0] ],
#     :μ₁ => [ [4., 3] ],
#     :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
#     :Σ₁ => [ [.5 0.1; 0.1 .2] ],
#     :μ_prior => [ zeros(3) ],
#     :Σ_prior => [ I(3) ],
#     :μ_initial => [ [1., 1, 1] ],
#     :Σ_initial => [ I(3) ],
#     :random_seed => [ 0 ],
# )

# ALG_PARAMS = Dict(
#     :update_method => [ :forward_euler ],
#     :kernel_cb => [ median_trick_cb! ],
#     :step_size => [ 0.0001 ],
#     :n_iter => [ 5000 ],
#     :n_particles => [ 10, 25, 50, 100 ],
#     :n_runs => [ 10 ],
#     :dKL_estimator => [ :RKHS_norm ],
#     :progress => [ false ],
#     )

###### Cell ###### - load data from the reruns
all_data10 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_10"))
all_data25 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_25"))
all_data50 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_50"))
all_data100 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_100"))
all_data = vcat(all_data10, all_data25, all_data50, all_data100)

for d in all_data
    display(plot_convergence(d))
    show_params(d)
    readline()
end
# Note that sometimes the flow estimate for logZ start lower than the lower bound.
# This is caused by the fact that the value used for 𝔼[V] is a simple MC estimate
# that is computed separately for each run and again for the plot.

###### Cell ###### - log Z estimates
# 50 particles:
[(mean(d[:estimated_logZ]), d[:Laplace_start]) for d in all_data50]
# (-13.096600752604735, 1)
# (-15.909862444306304, 0)
[(std(d[:estimated_logZ]), d[:Laplace_start]) for d in all_data50]
# (0.07727404426636961, 1)
# (9.03507264249623, 0)

###### Cell ###### - plot MAP v Laplace
mkpath(joinpath(plotdir, "MAPvLaplace50_5000iter"))

for (d, n) in zip(all_data50, ["Laplace", "MAP"])
    plot_convergence(d)
    saveplot("MAPvLaplace50_5000iter", n*".png")
end

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
