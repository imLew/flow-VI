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
using Examples

target_plotdir = "Adam"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)
mkpath(plotdir)
saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### - Load Data
all_data = load_data("bayesian_logistic_regression/Adam")

for d in all_data
    d[:true_logZ] = -13.34
    d[:EV] = 47.707
end

###### Cell ###### -
for d in all_data
    display(plot_convergence(d))
    @info mean(d[:estimated_logZ])
    show_params(d)
    readline()
end

###### Cell ###### -
Adam_data = filter_by_dict(Dict(:update_method => [:scalar_Adam]), all_data);
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
[std(d[:estimated_logZ]) for d in [f_data..., data...]]

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

###### Cell ###### -

###### Cell ###### -
