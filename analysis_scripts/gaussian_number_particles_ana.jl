###### Cell ###### - load dependencies
using DrWatson; @quickactivate
using BSON
using Distributions
using Plots
using StatsPlots
using KernelFunctions
using ValueHistories
using LinearAlgebra
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind;

using Utils

target_plotdir = "gauss_particles"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)

mkpath(plotdir)

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### -
# PROBLEM_PARAMS = Dict(
#     :problem_type => :gauss_to_gauss,
#     :μ₀ => [[0., 0]],
#     :μₚ => [[0, 0]],
#     :Σₚ => [[1. 0; 0 1.]],
#     :Σ₀ => [ 0.1*I(2), 10.0*I(2), ],
#     :random_seed => 0,
# )

# ALG_PARAMS = Dict(
#     :dKL_estimator => :RKHS_norm,
#     :n_iter => 4000,
#     :step_size => [ 0.01, 0.001, 0.005 ],
#     :n_particles => [50, 100, 200],
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :n_runs => 10,
# )

all_data = load_data("gaussian_to_gaussian/number_particles")
all_data = filter_by_dict( Dict(:n_iter => [10000]), all_data)

###### Cell ###### - plots by step size
data = filter_by_dict( Dict(:step_size => [0.001]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end

data = filter_by_dict(Dict(:step_size => [0.005]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end

data = filter_by_dict( Dict(:step_size => [0.01]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end

###### Cell ###### -
data = filter_by_dict( Dict(:n_particles=>[50], :step_size=>[0.005]), all_data)
vars = []
for d in data
    v = var(d[:estimated_logZ])
    rel_err = abs(d[:true_logZ]-mean(d[:estimated_logZ]))
    push!(vars, [d[:Σ₀], d[:n_iter], v, rel_err])
end

###### Cell ###### - boxplots 10I
# function compare_by_iter(step_size, n_particles, cov_fac, data; kwargs...)
#     data = filter_by_dict( Dict(:Σ₀=>[cov_fac*I(2)]), data)
#     data = filter_by_dict( Dict(:step_size=>[step_size]), data)
#     data = filter_by_dict( Dict(:n_particles=>[n_particles]), data)
#     sort!(data, by=d->d[:n_iter])
#     labels = [join(["$(d[key])" for key in [:n_iter]], "; ") for d in data]
#     make_boxplots(data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
#                   title="$n_particles"; kwargs...)
# end

###### Cell ###### -
function compare_by_particles(step_size, n_iter, cov_fac, data; kwargs...)
    data = filter_by_dict( Dict(:Σ₀=>[cov_fac*I(2)]), data)
    data = filter_by_dict( Dict(:step_size=>[step_size]), data)
    data = filter_by_dict( Dict(:n_iter=>[n_iter]), data)
    sort!(data, by=d->d[:n_particles])
    labels = ["$(d[:n_particles])" for d in data]
    bplt = make_boxplots(data, xticks=(1:9, labels), xrotation=60,
                         ylabel="log Z", colour=[colors[1] colors[2] colors[3]]; kwargs...)
    plt = plot();
    for (i, d) in enumerate(data)
        plot_integration!(plt, d, int_color=colors[i], flow_label=labels[i],
                          ylims=ylims(bplt));
    end
    return plot(plt, bplt, layout=grid(1,2, widths=[0.7, 0.3]))
end

###### Cell ###### - step size
function compare_by_stepsize(n_particles, n_iter, cov_fac, data; kwargs...)
    data = filter_by_dict( Dict(:Σ₀=>[cov_fac*I(2)]), data)
    data = filter_by_dict( Dict(:n_particles=>[n_particles]), data)
    data = filter_by_dict( Dict(:n_iter=>[n_iter]), data)
    sort!(data, by=d->d[:step_size])
    labels = ["$(d[:step_size])" for d in data]
    bplt = make_boxplots(data, xticks=(1:9, labels), xrotation=60,
                         ylabel="log Z", colour=[colors[1] colors[2] colors[3]]; kwargs...)
    plt = plot();
    for (i, d) in enumerate(data)
        plot_integration!(plt, d, int_color=colors[i], flow_label=labels[i],
                          ylims=ylims(bplt));
    end
    return plot(plt, bplt, layout=grid(1,2, widths=[0.7, 0.3]))
end

###### Cell ###### -
compare_by_stepsize(50, 10000, 0.1, all_data, ylims=(-5,5))
saveplot("narrow_stepsize.png")
compare_by_stepsize(50, 10000, 10.0, all_data, ylims=(-5,5))
saveplot("wide_stepsize.png")

###### Cell ###### - plots for number of particles
compare_by_particles(0.01, 10000, 0.1, all_data)
saveplot("narrow_particles.png")
compare_by_particles(0.01, 10000, 10.0, all_data)
saveplot("wide_particles.png")

###### Cell ###### -
