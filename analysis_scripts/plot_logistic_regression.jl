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

###### Cell ###### -
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
all_data = load_data(datadir("bayesian_logistic_regression", "MAPvLaplacevNormal"))
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
    @show d[:step_size]
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
    Plots.savefig(joinpath(plotdir, get_savename(d))*".png")
    # display(plt)
    # readline()
end
