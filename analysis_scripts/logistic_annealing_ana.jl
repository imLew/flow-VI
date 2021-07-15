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

target_plotdir = "logistic_annealing"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)
mkpath(plotdir)
saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### - Load Data
# ALG_PARAMS = Dict(
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :step_size => [ 0.001, 0.0001 ],
#     :n_iter => 10000,
#     :n_particles => 50,
#     :n_runs => 10,
#     :dKL_estimator => :RKHS_norm,
#     :annealing_schedule => [nothing, hyperbolic_annealing],
#     :annealing_params => Dict(:duration=>0.8, :p=>12),
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

###### Cell ###### -
all_data = load_data("bayesian_logistic_regression/annealing")

for d in all_data
    d[:EV] = 47.707
    d[:true_logZ] = -13.34
end

###### Cell ###### -
d = all_data[2];
plot_convergence(d)
show_params(d)

data = all_data[[1,2,4]]

###### Cell ###### -
data = filter_by_dict(Dict(:n_iter => [20000]), all_data)

plt = plot(legend=:bottomright, ylims=(-51, 1.7),
           xticks=(0:5000:20000, ["0", "5000", "10000", "15000", "20000"]))
labels = ["ϵ=0.001", "A-SVGD", "ϵ=0.0001"]
for i in 1:length(data)
    plot_integration!(plt, data[i], int_color=colors[i], flow_label=labels[i],
                 show_ribbon=false, ylims=(-51, 1.7),
                 xticks=(0:5000:20000, ["0", "5000", "10000", "15000", "20000"]),
                )
end
display(plt)
plot(plt, make_boxplots(data, yticks=nothing, ylims=(-51, 1.7), labels=labels,
                        legend=:none),
     layout=grid(1,2, widths=[0.7, 0.3]))

saveplot("AnnealedvsEuler.png")

# ylims(plt)  # (-50.80896439437238, -12.821591985386869)
# ylims(make_boxplots(data))  # (-44.76282353343334, 1.6170925009068293)

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

