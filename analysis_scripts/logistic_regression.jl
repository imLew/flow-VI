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

###### Cell ###### - Calculate EV once with many samples
# R = []
# for N in [1e5, 1e6, 1e7]
#     r = []
#     Threads.@threads for _ in 1:10
#         push!(r, expectation_V(all_data[1], n_samples=N))
#     end
#     push!(R, (N, mean(r), std(r)))
# end
# (100000.0, 47.64105027908237, 0.2664756245261788)
# (1.0e6, 47.684423426371126, 0.05245877064606076)
# (1.0e7, 47.69828643512527, 0.019409468940779617)

# EV = expectation_V(all_data[1], n_samples=1e8)
# 47.70672145899105

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
    d[:EV] = 47.707
end

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
#     )

###### Cell ###### - load data from the reruns
all_data10 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_10"))
all_data25 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_25"))
all_data50 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_50"))
all_data100 = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_100"))
all_data = vcat(all_data10, all_data25, all_data50, all_data100)

for d in all_data
    d[:EV] = 47.707
end

for d in all_data50
    d[:EV] = 47.707
end

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
    d[:EV] = 47.707
end

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

###### Cell ###### - plot RMSprop
mkpath(joinpath(plotdir, "RMSprop"))
plt = plot(legend=:bottomright)
plot_integration!(plt, data[1], int_color=colors[1], flow_label="ϵ=0.001",
                 show_ribbon=false)
plot_integration!(plt, data[end-2], int_color=colors[2], flow_label="RMSprop",
                 show_ribbon=false)
plot_integration!(plt, data[end], int_color=colors[3], flow_label="ϵ=0.0001",
                 show_ribbon=false)
saveplot("RMSprop", "RMSvEuler.png")

# Adam experiments
###### Cell ###### - params
# ALG_PARAMS = Dict(
#     :update_method => [ :scalar_Adam, :forward_euler ],
#     :β₁ => @onlyif(:update_method == :scalar_Adam, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
#     :β₂ => @onlyif(:update_method == :scalar_Adam, [0.9, 0.99, 0.999, 0.9999]),
#     :kernel_cb => median_trick_cb!,
#     :step_size => [ 0.001, @onlyif(:update_method == :forward_euler, 0.0001) ],
#     :n_iter => [ 10000 ],
#     :n_particles => 50,
#     :n_runs => 10,
#     :dKL_estimator => :RKHS_norm,
#     :Adam_unbiased => true,
#     :adam_stepsize_method => :minimum,
#     )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :logistic_regression,
#     :MAP_start => [ true ],
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

###### Cell ###### -
all_data = load_data(gdatadir("bayesian_logistic_regression", "GDvariants", "Adam"))

for d in all_data
    d[:true_logZ] = -13.34
    d[:EV] = 47.707
end

for d in all_data
    display(plot_integration(d))
    @info mean(d[:estimated_logZ])
    show_params(d)
    readline()
end

Adam_data = filter_by_dict( Dict(:update_method => [:scalar_Adam]), all_data )

###### Cell ###### - only with largest β₂
data = filter_by_dict( Dict(:β₂ => [0.9999]), Adam_data)

for d in data
    display(plot_integration(d))
    @info mean(d[:estimated_logZ])
    show_params(d)
    readline()
end

###### Cell ###### -
sort([(mean(d[:estimated_logZ]), d[:β₁], d[:β₂] ) for d in Adam_data],
     by=x->x[1])
# Overall, it seems that the best (in this case largest) values are obtained
# when using a small β₁ (0.7 or 0.8) and a large β₂ (0.999 or 0.9999).
# If we ignore the influence of momentum, which is was to gauge, then reducing
# β₁ reduces the step size. Conversely for β₂ a larger value means a smaller
# step size. These results fit that explaination, whether it is the whole picture
# is hard to tell.

sort([(std(d[:estimated_logZ]), d[:β₁], d[:β₂] ) for d in Adam_data],
     by=x->x[1])
# The variance also is also smallest for high values of β₂ and small values of
# β₁.

###### Cell ###### - plots
mkpath(joinpath(plotdir, "Adam"))

f_data = filter_by_dict(Dict(:update_method => [:forward_euler]), all_data);

###### Cell ###### - integration plots
data = filter_by_dict( Dict(:β₂ => [.999], :β₁ => [0.9, 0.7]), Adam_data);
plt = plot(legend=:bottomright)
cs = [colors[1], colors[3], colors[2], colors[4], colors[5], colors[6]]
ls = ["ϵ=0.001", "ϵ=0.0001"]
for (i,d) in enumerate([f_data..., data...])
    if d[:update_method] == :forward_euler
        l = "ϵ=$(d[:step_size])"
    else
        l = "β₁=$(d[:β₁]); β₂=$(d[:β₂])"
    end
    plot_integration!(plt, d, show_ribbon=false, int_color=cs[i], flow_label=l)
end
display(plt)
saveplot("Adam", "AdamVsEuler_integration.png")

[mean(d[:estimated_logZ]) for d in [f_data..., data...]]
# -14.743697837973286
# -16.796092050326966
# -15.680187476943507
# -15.910731222712382
[std(d[:estimated_logZ]) for d in [f_data..., data...]]
# 7.624474973483471
# 7.490476676808474
# 7.706374458619078
# 7.732743842071427

###### Cell ###### - estimation plots
plt = plot(legend=:outerright)
for β in [0.9, 0.99, 0.999, 0.9999]
    data = filter_by_dict(Dict(:β₂ => [β]), Adam_data);
    xy = [(d[:β₁], mean(d[:estimated_logZ]), std(d[:estimated_logZ])) for d in data]
    sort!(xy, by=xye->xye[2])
    scatter!(plt, [(x[1],x[2]) for x in xy], yerr=std(xy[3]), label="β₂=$β",
             alpha=0.6, markersize=3, xlabel="β₁", ylabel="log Z")
end
hline!(plt, [-13.34], label="true value")
for d in f_data
    hline!(plt, [mean(d[:estimated_logZ])], label="ϵ=$(d[:step_size])",
           ls=:dash)
end
hline!(plt,
      [entropy(MvNormal(data[1][:μ_initial], data[1][:Σ_initial]))-data[1][:EV]],
       label="H₀-𝔼[V]", ls=:dot)
display(plt)
saveplot("Adam", "AdamVsEuler.png")

###### Cell ###### -
plt = plot(legend=:outerright, ylabel="step size", xlabel="iterations")
cs = [colors[1], colors[2], colors[2], colors[4], colors[5], colors[6]]
for (i, β) in enumerate([0.9, 0.99, 0.999, 0.9999])
    data = filter_by_dict(Dict(:β₂ => [β]), Adam_data);
    xy = [mean([get(h, :step_sizes)[2] for h in d[:svgd_hist]]) for d in data]
    plot!(plt, xy, lw=0.1, color=cs[i],
          label=["β₂=$β" nothing nothing nothing nothing nothing nothing nothing nothing nothing nothing])
end
hline!(plt, [0.0001], ls=:dash, color=cs[5], label=nothing)
hline!(plt, [0.001], ls=:dash, color=cs[6], label=nothing)
display(plt)
saveplot("Adam", "Adam_stepsizes.png")

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
