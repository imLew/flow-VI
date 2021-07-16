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

target_plotdir = "WAG_WNes"
plotdir = joinpath( "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)
mkpath(plotdir)
saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### - Load Data
# ALG_PARAMS = Dict(
#     :step_size =>  0.001,
#     :n_iter =>  20000,
#     :n_particles =>  50,
#     :update_method => [:WAG, :WNES, :forward_euler],
#     :α => @onlyif(:update_method==:WAG, [π, 42]),
#     :c₁ => @onlyif(:update_method==:WNES, [0.1, 0.5, 1.0, 2.]),
#     :c₂ => @onlyif(:update_method==:WNES, [1, 0]),
#     :kernel_cb => median_trick_cb!,
#     :n_runs => 10,
#     :random_seed => 0,
# )

# PROBLEM_PARAMS = Dict(
#     :problem_type => :gauss_mixture_sampling,
#     :n_dim => 2,
#     :μ_initial => [ [0., 0.] ],
#     :Σ_initial => [ [1. 0; 0 1.], ],
#     :μₚ => [ [[8., -2.], [1., -6.], [4., -6.], [2., -3.]] ],
#     :Σₚ => [ [I(2), I(2), I(2), I(2), I(2)] ],
#     )

all_data = load_data("WAG_WNes")

d = all_data[1]

###### Cell ###### - Check how variable Δq is over the runs
WNes_data = filter_by_dict(Dict(:update_method=>[:WNES]), all_data)
WNes_data = filter_by_dict(Dict(:c₂=>[0,2]), WNes_data)
sort!(WNes_data, by=d->d[:c₂])
WAG_data = filter_by_dict(Dict(:update_method=>[:WAG]), all_data)
f_data = filter_by_dict(Dict(:update_method=>[:forward_euler]), all_data)

data = [WNes_data... f_data... WAG_data...]

failed = [d for d in data if any(isnan.(d[:svgd_results][1]))]
data = [d for d in data if !any(isnan.(d[:svgd_results][1]))]

for d in data
    F = [get(h, :ϕ_norm)[2] for h ∈ d[:svgd_hist]]
    @info maximum(std(F))
    show_params(d)
end

###### Cell ###### -
plt = plot(xticks=(0:5000:20000, ["$s" for s in 0:5000:20000]),
          ylabel="||Δq||")
for d in data
    F = [get(h, :ϕ_norm)[2] for h ∈ d[:svgd_hist]]
    # display(plot(mean(F), title=String(d[:update_method])))
    # readline()
    if d[:update_method] == :forward_euler
        c = colors[1]
    elseif d[:update_method] == :WAG
        c = colors[2]
    elseif d[:update_method] == :WNES
        c = colors[3]
    end
    if d[:update_method] == :WAG
        l = "α=$(round(d[:α], digits=2))"
        ls=:solid
    elseif d[:update_method] == :WNES
        l = "c₁=$(d[:c₁]), c₂=$(d[:c₂])"
        if d[:c₂] == 2
            ls=:dash
        else
            ls=:dashdot
        end
    elseif d[:update_method] == :forward_euler
        l = "SVGD"
        ls=:dot
    end
    plot!(plt, mean(F), ribbon=std(F), label=l, ls=ls)
end
display(plt)
saveplot("accelerated_steps.png")

###### Cell ###### -
d = all_data[2]
F = plot([get(h, :ϕ_norm)[2] for h ∈ d[:svgd_hist]])
show_params(d)

###### Cell ###### -
WAG = filter_by_dict(Dict(:update_method => [:WAG]), data)
WNES = filter_by_dict(Dict(:update_method => [:WNES]), data)
forward_euler = filter_by_dict(Dict(:update_method => [:forward_euler]), data)

# plt = plot()
for d in forward_euler
    F = [get(h, :ϕ_norm)[2] for h ∈ d[:svgd_hist]]
    display(plot(F))
    readline()
end

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

