###### Cell ###### - load dependencies
using DrWatson
@quickactivate
using Plots
using ValueHistories
using BSON
using KernelFunctions
using Distributions
using LinearAlgebra
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind
# using PDMats

using Utils

target_plotdir = "RMSprop"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)
mkpath(plotdir)
saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### - Load Data
# ALG_PARAMS = Dict(
#     :update_method => [ :forward_euler, :scalar_RMS_prop ],
#     :γ => @onlyif(:update_method==:scalar_RMS_prop, collect(0.1:0.1:1)),
#     :kernel_cb => median_trick_cb!,
#     :step_size => [ 0.001, @onlyif(:update_method==:forward_euler, 0.0001) ],
#     :n_iter => 20000,
#     :n_particles => 50,
#     :n_runs => 10,
#     :dKL_estimator => :RKHS_norm,
#     )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :logistic_regression,
#     :MAP_start => true,
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
all_data = load_data(gdatadir("bayesian_logistic_regression/RMSprop"))

for d in all_data
    d[:EV] = 47.707
    d[:true_logZ] = -13.34
end

###### Cell ###### -
data = all_data

plt = plot(legend=:bottomright, xticks=(0:5000:20000, ["$s" for s in 0:5000:20000]))
plot_integration!(plt, data[1], int_color=colors[1], flow_label="ϵ=0.001",
                 show_ribbon=false)
plot_integration!(plt, data[end-2], int_color=colors[2], flow_label="RMSprop",
                 show_ribbon=false)
plot_integration!(plt, data[end], int_color=colors[3], flow_label="ϵ=0.0001",
                 show_ribbon=false)
display(plt)
saveplot("RMSvEuler.png")

###### Cell ###### -
d = all_data[11];
plot_integration(d)
show_params(d)


###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

