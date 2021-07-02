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

# RMS prop experiments
###### Cell ###### - params
# ALG_PARAMS = Dict(
#     :update_method => [ :forward_euler, :scalar_RMS_prop, ],
#     :β₁ => 0.9,
#     :β₂ => 0.999,
#     :γ => @onlyif(:update_method == :scalar_RMS_prop, [ 0.7, 0.8, 0.9, 0.95 ]),
#     :kernel_cb => median_trick_cb!,
#     :step_size => [ @onlyif(:update_method == :scalar_RMS_prop, 0.001),
#                    @onlyif(:update_method == :forward_euler, 0.0001) ],
#     :n_iter => [ 5000 ],
#     :n_particles => 50,
#     :n_runs => 10,
#     :dKL_estimator => :RKHS_norm,
#     :progress => false,
#     )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :logistic_regression,
#     :MAP_start => [ true ],
#     :Laplace_start => [ false,  ],
#     :n_dim => 2,
#     :n₀ => 50,
#     :n₁ => 50,
#     :μ₀ => [ [0., 0] ],
#     :μ₁ => [ [4., 3] ],
#     :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
#     :Σ₁ => [ [.5 0.1; 0.1 .2] ],
#     :μ_prior => [ zeros(3) ],
#     :Σ_prior => [ I(3) ],
#     :μ_initial => [ [1., 1, 1] ],
#     :Σ_initial => [ I(3) ],
#     :random_seed => 0,
# )

###### Cell ###### - load data and plot for overview
all_data = load_data(gdatadir("bayesian_logistic_regression", "GDvariants", "RMSprop"))

for d in all_data
    d[:true_logZ] = -13.34
end

data = filter_by_dict( Dict(:n_iter => [10000]), all_data )

for d in data
    display(plot_integration(d))
    @info mean(d[:estimated_logZ])
    show_params(d)
    readline()
end
# interestingly the value of γ seems to have very little influence on the final
# log Z estimate

###### Cell ###### - compute mean+minimum step size and log Z mean+std
[ mean([mean(get(h, :step_sizes)[2]) for h in d[:svgd_hist]]) for d in data ]
# 0.001000000
# 0.000411451
# 0.000413492
# 0.000414547
# 0.000414912
# 0.000415101
# 0.000415217
# 0.000415297
# 0.000415355
# 0.000415399
# 0.000415435
# 0.000100000

[minimum([minimum(get(h, :step_sizes)[2]) for h in d[:svgd_hist]]) for d in data]
# 0.001
# 1.4532184809476856e-5
# 1.4281094177668533e-5
# 1.4134689509161302e-5
# 1.4080990216459099e-5
# 1.4052501409510837e-5
# 1.403439527177852e-5
# 1.4021742130566222e-5
# 1.4011850883448029e-5
# 1.4003924958098671e-5
# 1.399730741532191e-5
# 0.0001

[(mean(d[:estimated_logZ]), get(d, :γ, 0)) for d in data]
# (-12.768765317862108, 0)
# (-13.283954727675745, 0.05)
# (-13.278882521989038, 0.1)
# (-13.276283035796519, 0.2)
# (-13.275386723793272, 0.3)
# (-13.274925636769378, 0.4)
# (-13.274641877501981, 0.5)
# (-13.27444822539806, 0.6)
# (-13.274306787673115, 0.7)
# (-13.274198393691767, 0.8)
# (-13.27411224170991, 0.9)
# (-14.872155551080146, 0)
[(std(d[:estimated_logZ]), get(d, :γ, 0)) for d in data]
# (9.181622680897695, 0)
# (9.050941587655766, 0.05)
# (9.052462710563525, 0.1)
# (9.053240466362784, 0.2)
# (9.053507552058575, 0.3)
# (9.053644449982988, 0.4)
# (9.053728419476416, 0.5)
# (9.053785551268547, 0.6)
# (9.053827157176023, 0.7)
# (9.05385894644824, 0.8)
# (9.053884147686995, 0.9)
# (9.155967448673595, 0)

###### Cell ###### -
mkpath(joinpath(plotdir, "RMSprop"))
plt = plot(legend=:bottomright)
plot_integration!(plt, data[1], int_color=colors[1], flow_label="ϵ=0.001",
                 show_ribbon=false)
plot_integration!(plt, data[end-2], int_color=colors[2], flow_label="RMSprop",
                 show_ribbon=false)
plot_integration!(plt, data[end], int_color=colors[3], flow_label="ϵ=0.0001",
                 show_ribbon=false)
saveplot("RMSprop", "RMSvEuler.png")

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
