###### Cell ###### - load dependencies
using DrWatson
@quickactivate
using Plots
using StatsPlots
using ValueHistories
using BSON
using KernelFunctions
using LinearAlgebra
using Distributions
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind
# using PDMats

using Utils
# using Examples
# LogReg = LogisticRegression
# Lin = LinearRegression

target_plotdir = "annealing"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

mkpath(plotdir)

###### Cell ###### - Load data
# ALG_PARAMS = Dict(
#     :step_size =>  0.01,
#     :n_iter =>  20000,
#     :n_particles =>  50,
#     :dKL_estimator => :RKHS_norm,
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :n_runs => 10,
#     :random_seed => 0,
#     :annealing_schedule => [linear_annealing, hyperbolic_annealing,
#                             cyclic_annealing],
#     :annealing_params => [
#                       @onlyif(:annealing_schedule == linear_annealing,
#                               Dict(:duration=>0.6)),
#                       @onlyif(:annealing_schedule == hyperbolic_annealing,
#                               Dict(:duration=>0.6, :p=>8), ),
#                       @onlyif(:annealing_schedule == hyperbolic_annealing,
#                               Dict(:duration=>0.8, :p=>8), ),
#                       @onlyif(:annealing_schedule == hyperbolic_annealing,
#                               Dict(:duration=>0.6, :p=>12)),
#                       @onlyif(:annealing_schedule == hyperbolic_annealing,
#                               Dict(:duration=>0.8, :p=>12)),
#                       @onlyif(:annealing_schedule == cyclic_annealing,
#                               Dict(:duration=>0.6, :p=>12, :C=>3)),
#                          ],
# )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :gauss_mixture_sampling,
#     :n_dim => 2,
#     :μ_initial => [ [0., 0.] ],
#     :Σ_initial => [ [1. 0; 0 1.], ],
#     :μₚ => [ [[-8., -2.], [1., 6.], [2., -1.]] ],
#     :Σₚ => [ [[1. 0.5; 0.5 1], [1.2 0.1; 0.1 1.2], I(2)] ],
#     )
all_data = load_data("gaussian_mixture_sampling/annealing");

###### Cell ###### -
for d in all_data
    # @info mean(d[:estimated_logZ])
    show_params(d)
    # display(plot_convergence(d))
    readline()
end

###### Cell ###### - plots of annealing schedules
plt = plot_annealing_schedule(all_data[2], label="Hyperbolic",
                              xticks=(0:5000:20000,
                                      ["0", "5000", "10000", "15000", "20000"]),
                              legend=:bottomright, ylabel="γ(t)", xlabel="t")
plot_annealing_schedule!(plt, all_data[5], label="Cyclic")
plot_annealing_schedule!(plt, all_data[6], label="Linear")
all_data[2][:annealing_params]
all_data[5][:annealing_params]
all_data[6][:annealing_params]
saveplot("annealing_schedules.png")

###### Cell ###### - boxplots of log Z estimation
labels = []
for d in all_data
    if haskey(d, :annealing_params)
        p = d[:annealing_params]
        x = ["D=$(p[:duration])"]
        haskey(p, :p) ? push!(x, "p=$(p[:p])") : nothing
        haskey(p, :C) ? push!(x, "C=$(p[:C])") : nothing
        push!(labels, Tuple(x))
    else
        push!(labels, ("SVGD",))
    end
end

make_boxplots(all_data, size=(800,500),
              xticks=(1:length(labels), [join(s, ", ") for s in labels]),
              xrotation=60, ylabel="log Z")
saveplot("annealing_results_boxplot.png")

###### Cell ###### - load data for standard repeat experiment
# ALG_PARAMS = Dict(
#     :step_size =>  0.01,
#     :n_iter =>  20000,
#     :n_particles =>  50,
#     :dKL_estimator => :RKHS_norm,
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :n_runs => 10,
#     :random_seed => 0,
#     :annealing_schedule => [nothing, hyperbolic_annealing],
#     :annealing_params => Dict(:duration=>0.8, :p=>12),
# )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :gauss_mixture_sampling,
#     :n_dim => 2,
#     :μ_initial => [ [0., 0.] ],
#     :Σ_initial => [ [1. 0; 0 1.], ],
#     :μₚ => [ [[8., -2.], [1., -6.], [4., -6.], [2., -3.]],
#               [[2., -2.], [3., -6.], [-4., -6.], [2., -3.]]],
#     :Σₚ => [ [I(2), I(2), I(2), I(2), I(2)] ],
#     )
# From the data above it appears as though standard SVGD is actually much better
# at estimating log Z of a mixture than annealing SVGD. This is unexpected
# since standard fails to fail
fu_data = load_data("gaussian_mixture_sampling/vanilla_test");

###### Cell ###### - make boxplot for follow up data
labels = []
for d in fu_data
    if haskey(d, :annealing_params)
        p = d[:annealing_params]
        x = ["D=$(p[:duration])"]
        haskey(p, :p) ? push!(x, "p=$(p[:p])") : nothing
        haskey(p, :C) ? push!(x, "C=$(p[:C])") : nothing
        push!(labels, Tuple(x))
    else
        push!(labels, ("SVGD",))
    end
end

make_boxplots(fu_data, size=(500,400),
              xticks=(1:length(labels), [join(s, ", ") for s in labels]),
              xrotation=60, ylabel="log Z")

###### Cell ###### - plot sampling results
# Create some plots to illustrate the effects of annealing
size = (250,200)
plot_2D_results(all_data[1], xticks=nothing, yticks=nothing, size=size)
saveplot("dist_no_annealing.png")
plot_2D_results(all_data[4], xticks=nothing, yticks=nothing, size=size)
saveplot("dist_hyper_annealing.png")
plot_2D_results(all_data[5], xticks=nothing, yticks=nothing, size=size)
saveplot("dist_cyclis_annealing.png")
show_params(all_data[5])
plot_2D_results(all_data[6], xticks=nothing, yticks=nothing, size=size)
saveplot("dist_linear_annealing.png")

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

