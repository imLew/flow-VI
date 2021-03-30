##Cell
using BSON
using Distributions
using Plots
using StatsPlots
using KernelFunctions
using ValueHistories
using LinearAlgebra
using DrWatson
using PrettyTables
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils

##Cell
all_data = [BSON.load(n) for n in readdir(datadir("gaussian_to_gaussian", "method_compare"), join=true) ];
all_data = [data for data in all_data if data[:failed_count]<10];
for d in all_data
    d[:svgd_hist] = convert(Array{MVHistory}, d[:svgd_hist])
end

##Cell
target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/gauss/method_compare/")
mkpath(plotdir)

##Cell
WNES_path = joinpath(plotdir, "WNES")
mkpath(WNES_path)
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ 10*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]
                                ), 
                          all_data)
    plt = make_boxplots(data, title="Σ=10*I, ϵ=$ϵ", 
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10))
    display(plt)
#    readline()
     Plots.savefig(joinpath(WNES_path, "10I_stepsize=$ϵ.png"))
end

##Cell
# For the larger covariance a small stepsize was best.
# The least variance in results was obtained with c₁ = c₂ = 0.1 and c₂ = 5, c₁ =-.1

## Cell
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ .1*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]), all_data)
    plt = make_boxplots(data, title="Σ=0.1*I, ϵ=$ϵ", 
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10))
    display(plt)
#    readline()
     Plots.savefig(joinpath(WNES_path, "0.1I_stepsize=$ϵ.png"))
end

##Cell
# for small covariance only c₁=c₂=0.1 was any good

##Cell
WAG_path = joinpath(plotdir, "WAG")
mkpath(WAG_path)
for C in [0.1, 10.]
    for ϵ in [0.5, 0.05, 0.005]
        data = filter_by_dict( Dict(:update_method => [:naive_WAG],
                                    :Σ₀ => [ C*[1.0 0; 0 1] ],
                                    :step_size => [ ϵ ]
                                    ), 
                              all_data)
        plt = make_boxplots(data, title="Σ=$(C)I, ϵ=$ϵ", 
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10))
        display(plt)
#        readline()
         Plots.savefig(joinpath(WAG_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

##Cell
# α=7 was the best across all settings if there was any difference
# except large for with ϵ=0.05, which was the best overall
# in general all the estimates seem shit and mostly with high variance
# at small step sizes nothing happened

##Cell
RMSprop_path = joinpath(plotdir, "RMSprop")
mkpath(RMSprop_path)
for C in [0.1, 10.]
    for ϵ in [0.5, 0.05, 0.005]
        data = filter_by_dict( Dict(:update_method => [:scalar_RMS_prop],
                                    :Σ₀ => [ C*[1.0 0; 0 1] ],
                                    :step_size => [ ϵ ]
                                    ), 
                              all_data)
        plt = make_boxplots(data, title="Σ=$(C)I ϵ=$ϵ", size=(300,200),
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10))
        display(plt)
#        readline()
        Plots.savefig(joinpath(RMSprop_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

##Cell
# for small covariance γ=0.8 was always good while γ=0.9 was only good with ϵ=0.005
# which large cov γ=0.8 was also consistently better (except for the case ϵ=0.005)
# where the again performed equally well but not very good overall 
# (possible they just didn't converge)

##Cell
Adam_path = joinpath(plotdir, "Adam")
mkpath(Adam_path)
for C in [0.1, 10.]
    data = filter_by_dict( Dict(:update_method => [:scalar_Adam],
                                :Σ₀ => [ C*[1.0 0; 0 1] ],
                               ), 
                          all_data)
    plt = make_boxplots(data, title="Σ=$(C)I", 
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        labels = ["ϵ = $(d[:step_size])" for d in data])
    display(plt)
   # readline()
   Plots.savefig(joinpath(Adam_path, "$(C)I.png"))
end

##Cell
# The smallest step size gave the best results

##







## ## table_file = joinpath(target_root, "gaussian_covariance_table_1_2.tex")
## open(table_file, "w") do io
##     pretty_table(io, 
##      hcat([ d[:failed_count]==0 ? round.(d[:estimated_logZ], digits=2) : [round.(d[:estimated_logZ], digits=2); NaN] for d in all_data[1:9] ]...),
##      hcat([ [d[:Σ₀], d[:update_method]] for d in [d for d in all_data[1:9]] ]...),
##      backend=:latex
##     )
## end

## table_file = joinpath(target_root, "gaussian_covariance_table_2_2.tex")
## open(table_file, "w") do io
##     pretty_table(io, 
##      hcat([ d[:failed_count]==0 ? round.(d[:estimated_logZ], digits=2) : [round.(d[:estimated_logZ], digits=2); NaN] for d in all_data[10:end] ]...),
##      hcat([ [d[:Σ₀], d[:update_method]] for d in [d for d in all_data[10:end]] ]...),
##      backend=:latex
##     )
## end
