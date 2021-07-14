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

target_plotdir = "bayesian_logistic_regression"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)
mkpath(plotdir)

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

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

###### Cell ###### - Load Data
# ALG_PARAMS = Dict(
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :step_size => 0.0001,
#     :n_iter => 10000,
#     :n_particles => 50,
#     :n_runs => 10,
#     :dKL_estimator => :RKHS_norm,
#     )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :logistic_regression,
#     :MAP_start => [  true, false ],
#     :Laplace_start => [ false,  @onlyif(:MAP_start==true, true) ],
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

all_data = load_data(gdatadir("bayesian_logistic_regression", "laplace"))

# ###### Cell ###### -
# for d in all_data
#     if d[:Laplace_start] || !d[:MAP_start]
#         d[:EV] = expectation_V(d, n_samples=1e8)
#     else
#         d[:EV] = 47.707
#     end
#     d[:true_logZ] = -13.34
# end
[d[:EV] for d in all_data]
H = entropy(MvNormal(all_data[1][:μ_initial], all_data[1][:Σ_initial]))
H - all_data[1][:EV]

###### Cell ###### -
for (d, n) in zip(all_data, ["Laplace", "MAP", "Normal"])
    display(plot_convergence(d, size=(400,300)))
    # readline()
    saveplot("MAPvLaplacevNormal"*n*".png")
end

###### Cell ###### -
[mean(d[:estimated_logZ]) for d in all_data]
[std(d[:estimated_logZ]) for d in all_data]
# -12.9868197385573 ± 0.112311802834092
# -15.0448901437442 ± 7.500020747103735
# -33.5682559728682 ± 29.26002386060209

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
