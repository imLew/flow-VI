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

###### Cell ###### - estimation plots
# size(d[:estimated_logZ])
d = all_data[1]
est_logZ(d, n=10000) = [e[n] for e in estimate_logZ(d)]
mean(est_logZ(d))

###### Cell ###### - estimation plots
plt = plot(legend=:outerright)
# 0.2, 0.5, 0.8, 0.999,
for β in [0.0, 0.9999]
    data = filter_by_dict(Dict(:β₂ => [β]), Adam_data);
    xy = [(d[:β₁], mean(est_logZ(d)), std(est_logZ(d))) for d in data]
    sort!(xy, by=xye->xye[2])
    scatter!(plt, [(x[1],x[2]) for x in xy], yerr=std(xy[3]), label="β₂=$β",
             alpha=0.6, markersize=3, xlabel="β₁", ylabel="log Z")
end
hline!(plt, [-13.34], label="true value")
for d in f_data
    hline!(plt, [mean(est_logZ(d))], label="ϵ=$(d[:step_size])",
           ls=:dash)
end
hline!(plt,
      [entropy(MvNormal(data[1][:μ_initial], data[1][:Σ_initial]))-data[1][:EV]],
       label="H₀-𝔼[V]", ls=:dot)
display(plt)
# saveplot("AdamVsEuler.png")

# ###### Cell ###### -
# plt = plot(legend=:outerright, ylabel="step size", xlabel="iterations",
#            xticks=(0:5000:20000, ["0", "5000", "10000", "15000", "20000"]))
# cs = [colors[1], colors[2], colors[2], colors[4], colors[5], colors[6]]
# for (i, β) in enumerate([0.8, 0.999, 0.9999])
#     data = filter_by_dict(Dict(:β₂ => [β]), Adam_data);
#     xy = [mean([get(h, :step_sizes)[2] for h in d[:svgd_hist]]) for d in data]
#     plot!(plt, xy, lw=0.1, color=cs[i],
#           label=["β₂=$β" nothing nothing nothing nothing nothing nothing nothing nothing nothing nothing])
# end
# hline!(plt, [0.0001], ls=:dash, color=cs[5], label=nothing)
# hline!(plt, [0.001], ls=:dash, color=cs[6], label=nothing)
# display(plt)
# # saveplot("Adam_stepsizes.png")

# ###### Cell ###### -
# plt = plot(legend=:outerright, ylabel="step size", xlabel="iterations",
#            xticks=(0:5000:20000, ["0", "5000", "10000", "15000", "20000"]))
# cs = [colors[1], colors[2], colors[3], colors[4], colors[5], colors[6]]
# for (i, β) in enumerate([0.0, 0.2, 0.5, 0.8, 0.999, 0.9999])
#     data = filter_by_dict(Dict(:β₂ => [β]), Adam_data);
#     xy = [[get(h, :step_sizes)[2] for h in d[:svgd_hist]] for d in data]
#     plot!(plt, mean(xy[1]), lw=0.1, color=cs[i], label="β₂=$β")
#     for s in xy[2:end]
#         plot!(plt, mean(s), lw=0.1, color=cs[i], label=nothing)
#     end
# end
# hline!(plt, [0.0001], ls=:dash, color=cs[5], label=nothing)
# hline!(plt, [0.001], ls=:dash, color=cs[6], label=nothing)
# display(plt)
# # saveplot("Adam_stepsizes.png")

###### Cell ###### -
data = [f_data[1]]
da = filter_by_dict(Dict(:β₂=>[0.999], :β₁=>[0.9]), Adam_data)
push!(data, da[1])
da = filter_by_dict(Dict(:β₂=>[0.999], :β₁=>[0.0]), Adam_data)
push!(data, da[1])
da = filter_by_dict(Dict(:β₂=>[0.0], :β₁=>[0.9]), Adam_data)
push!(data, da[1])
push!(data, f_data[end])

plt = plot(legend=:bottomright, xticks=(0:5000:20000, ["$s" for s in 0:5000:20000]),
           ylims=(-45, 1.7))
labels = ["ϵ=0.001", "β₁=0.9, β₂=0.999", "β₁=0, β₂=0.999", "β₁=0.9, β₂=0", "ϵ=0.0001"]
for i in 1:length(data)
    plot_integration!(plt, data[i], int_color=colors[i], flow_label=labels[i],
                 show_ribbon=false,  ylims=(-45, 1.7),
                 xticks=(0:5000:20000, ["$s" for s in 0:5000:20000]),
                )
end
display(plt)
plot(plt, make_boxplots(data, yticks=nothing,  labels=labels, ylims=(-45, 1.7),
                        legend=:none),
     layout=grid(1,2, widths=[0.7, 0.3]))
saveplot("AdamVsEuler_integration.png")
# ylims(plt)  # (-43.45018440038598, -12.821591985386869)
# ylims(make_boxplots(data))  # (-44.76282353343334, 1.6170925009068293)

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
