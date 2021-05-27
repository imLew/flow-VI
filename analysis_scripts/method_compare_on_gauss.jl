###### Cell ###### - load dependencies
using DrWatson
using BSON
using Distributions
using Plots
using StatsPlots
using KernelFunctions
using ValueHistories
using LinearAlgebra
# using PrettyTables
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind;

# using SVGD
using Utils

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)

## Search for step size, #particles and #iter for following experiments
###### Cell ###### -

# PROBLEM_PARAMS = Dict(
#     :problem_type => [ :gauss_to_gauss ],
#     :μ₀ => [[0., 0]],
#     :μₚ => [[0, 0]],
#     :Σₚ => [[1. 0; 0 1.]],
#     :Σ₀ => [ 0.1*I(2), 10.0*I(2), ],
#     :random_seed => [ 0 ],
# )

# ALG_PARAMS = Dict(
#     :dKL_estimator => [ :RKHS_norm ],
#     :n_iter => [1000, 2000],
#     :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
#     :step_size => [ 0.05, 0.001, 0.005 ],
#     :n_particles => [50, 100, 200],
#     :update_method => [ :forward_euler ],
#     :kernel_cb => [median_trick_cb!],
#     :n_runs => 10,
# )

###### Cell ###### -
all_data = load_data( "gaussian_to_gaussian/initial_grid" )

###### Cell ###### -
data = filter_by_dict( Dict(:n_iter => [1000]), all_data)
for d in data
    show_params(d)
    display(plot_integration(d))
    readline()
end
# 0.001 and 100 did not converge
# 0.005 and 100 did not converge
# 0.05 and 100 overshot but didn't flatten out
# for 200 particles none of the step sizes converged
# except 0.05 with the narrow initial distribution
# for 50 particles 0.05 converged in both cases but the others still didn't

###### Cell ###### -
data = filter_by_dict( Dict(:step_size => [0.001]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end
# none of these look converged

data = filter_by_dict( Dict(:step_size => [0.005]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end
# 1000i; 100p look OK-ish for narrow initial
# same for 200p and 50p
# for 2000i
# 100p wide did not converge; narrow looks close
# same for 200p
# for 50p wide did not reach the target while narrow did and overshot without
# flatting

data = filter_by_dict( Dict(:step_size => [0.05]), all_data)
for d in data
    show_params(d)
    display(plot_convergence(d))
    readline()
end

###### Cell ###### -
###### Cell ###### -
###### Cell ###### -
###### Cell ###### -
###### Cell ###### -
###### Cell ###### -
## Cell  WNES vs WAG vs Euler - params

# PROBLEM_PARAMS = Dict(
#     :problem_type => [ :gauss_to_gauss ],
#     :μ₀ => [[0., 0]],
#     :μₚ => [[0, 0]],
#     :Σₚ => [[1. 0; 0 1.]],
#     :Σ₀ => [ 0.1*I(2), 10.0*I(2) ],
#     :random_seed => [ 0 ],
# )

# ALG_PARAMS = Dict(
#     :dKL_estimator => [ :RKHS_norm ],
#     :n_iter => [2000],
#     :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
#     :step_size => [0.05, 0.005],
#     :n_particles => [100],
#     :update_method => [:forward_euler, :naive_WAG, :naive_WNES],
#     :α => @onlyif(:update_method == :naive_WAG, [3.1, 4, 7] ),
#     :c₁ => @onlyif(:update_method == :naive_WNES, [.1, 1, .5] ),
#     :c₂ => @onlyif(:update_method == :naive_WNES, [.1, 1, .5] ),
#     :kernel_cb => [median_trick_cb!],
#     :n_runs => 10,
# )

###### Cell ###### - load data
plot_cb(args...; kwargs...) = nothing
for n in readdir(datadir("gaussian_to_gaussian", "WNESvWAGvEuler"), join=true)
    d = BSON.load(n)
    if d[:svgd_results]==[]
        rm(n)
    end
end
all_data = load_data("gaussian_to_gaussian", "WNESvWAGvEuler");

###### Cell ###### - set up target dir for plots
target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/gauss/WNESvWAGvEuler/")
mkpath(plotdir)

###### Cell ###### - plot everything to get an overview
for d in all_data
    display(plot_integration(d))
    show_params(d)
    if readline() == "q"
        break
    end
end

###### Cell ###### -- Σ=0.1 WNES boxplots
plt1, plt2 = plot(), plot()
data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.05],
                            :Σ₀ => [ 0.1*I(2) ],
                            :n_iter => [ 2000 ],
                           ),
                      all_data);
make_boxplots(data, legend_keys=[:c₁, :c₂])

keys = [:c₁, :c₂]
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt1, data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
               ylims=(0,4.5), title="ϵ = 0.05")

# the pattern in this plot is that WNES performs best if c₁ ≤ c₂, the plots
# for which this is true are the 5 that are clearly closest to the target
# it also slightly suggests that larger c₂ also helps (the three plots closest
# to the target have c₂=1 which was the largest value used in this experiment

data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.005],
                            :Σ₀ => [ 0.1*I(2) ],
                            :n_iter => [ 2000 ],
                           ),
                      all_data);
make_boxplots(data, legend_keys=[:c₁, :c₂])
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt2, data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
               ylims=(0,4.5), title="ϵ = 0.005")

# the same pattern exists with the smaller step size with the difference that
# the fit is now even closer to the true value

plot(plt1, plt2, size=(600,500))
#savefig(joinpath(plotdir, "WNES_S0.1.png"))

###### Cell ###### -- Σ=10 WNES boxplots
plt1, plt2 = plot(), plot()
data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.05],
                            :Σ₀ => [ 10*I(2) ],
                            :n_iter => [ 2000 ],
                           ),
                      all_data);

keys = [:c₁, :c₂]
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt1, data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10), title="ϵ = 0.05")

data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.005],
                            :Σ₀ => [ 10*I(2) ],
                            :n_iter => [ 2000 ],
                           ),
                      all_data);
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt2, data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10), title="ϵ = 0.005")

plot(plt1, plt2, size=(600,500))
#savefig(joinpath(plotdir, "WNES_S10.png"))

###### Cell ###### -- WNES Σ=10; ϵ=0.005 more iterations
plt1, plt2 = plot(), plot()
data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :Σ₀ => [ 0.1*I(2) ],
                            :n_iter => [ 4000 ],
                           ),
                      all_data);
keys = [:c₁, :c₂]
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt1, data, xticks=(1:9, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10), title="Σ = 0.1")

data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :Σ₀ => [ 10*I(2) ],
                            :n_iter => [ 4000 ],
                           ),
                      all_data);
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt2, data, xticks=(1:6, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10), title="Σ = 10")

plot(plt1, plt2, size=(600,500))
#savefig(joinpath(plotdir, "WNES_4000iter.png"))

###### Cell ###### -- Σ=10 WAG boxplots
keys = [:α]

plts = [ plot(), plot(), plot(), plot() ]
for (plt, eps) in zip( plts, [0.5, 0.1, 0.05, 0.005] )
    data = filter_by_dict( Dict( :update_method => [:naive_WAG],
                                :step_size => [eps],
                                :Σ₀ => [ 10*I(2) ],
                               ),
                          all_data);
    labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
    make_boxplots!(plt, data, xticks=(1:3, labels), xrotation=60, ylabel="log Z",
                   ylims=(-5,4), title="ϵ = $eps")
end

plot(plts..., size=(600,500))
#savefig(joinpath(plotdir, "WAG_S10.png"))

###### Cell ###### -- Σ=0.1 WAG boxplots
keys = [:α]

plts = [ plot(), plot(), plot(), plot() ]
for (plt, eps) in zip( plts, [0.5, 0.1, 0.05, 0.005] )
    data = filter_by_dict( Dict( :update_method => [:naive_WAG],
                                :step_size => [eps],
                                :Σ₀ => [ 0.1*I(2) ],
                               ),
                          all_data);
    labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
    make_boxplots!(plt, data, xticks=(1:3, labels), xrotation=60, ylabel="log Z",
                   ylims=(0,5.2), title="ϵ = $eps")
end

plot(plts..., size=(600,500))
#savefig(joinpath(plotdir, "WAG_S0.1.png"))

###### Cell ###### -- WAG integration
plts = []
data = filter_by_dict( Dict( :update_method => [:naive_WAG],
                            :Σ₀ => [ 10*I(2) ],
                           ),
                      all_data);
data = filter_by_dict( Dict( :α => [ 3.1 ],), data);
for d in data
    show_params(d)
    plt = plot_convergence(d)
    push!(plts, plt)
    display(plt)
    # readline()
end

plot(plts..., size=(800,700))
saveplot("WAGa=3.1integration.png")

###### Cell ###### -- vanilla SVGD
plt1, plt2 = plot(), plot()
data = filter_by_dict( Dict( :update_method => [:forward_euler],
                            # :step_size => [0.05],
                            :Σ₀ => [ 0.1*I(2) ],
                           ),
                      all_data);
keys = [:step_size]
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt1, data, xticks=(1:3, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10))

data = filter_by_dict( Dict( :update_method => [:forward_euler],
                            # :step_size => [0.005],
                            :Σ₀ => [ 10*I(2) ],
                           ),
                      all_data);
labels = [join(["$(key)=$(d[key])" for key in keys], "; ") for d in data]
make_boxplots!(plt2, data, xticks=(1:3, labels), xrotation=60, ylabel="log Z",
               ylims=(-5,10))

plot(plt1, plt2, size=(600,500))
#savefig(joinpath(plotdir, "WNES_S10.png"))

###### Cell ######
data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.05],
                            :Σ₀ => [ 10*I(2) ],
                           ),
                      all_data);
data = filter_by_dict( Dict( :c₁ => [ 1.0 ], :c₂ => [ 0.1 ]), data);
plot_integration(data[1])

###### Cell ######
data = filter_by_dict( Dict( :update_method => [:naive_WNES],
                            :step_size => [0.005],
                            :Σ₀ => [ 0.1*I(2) ],
                           ),
                      all_data);
make_boxplots(data, legend_keys=[:c₁, :c₂])








## # Comparison of different initial covariances

###### Cell ###### - load data
all_data = [BSON.load(n) for n in readdir(datadir("gaussian_to_gaussian", "covariance"), join=true) ];

###### Cell ###### - set up target dir for plots
target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/covariance_all/")

###### Cell ###### - boxplot estimation results by initial cov
for C in [0.001, 0.01, 0.1, 1, 10, 100]
    data = filter_by_dict( Dict(:Σ₀ => [ C*[1.0 0; 0 1] ]), all_data)

    plt = boxplot([d[:estimation_rkhs] for d in data],
                  labels=reshape(["$(d[:update_method])" for d in data],
                                 (1,length(data))),
                  legend=:outerright; title="Log Z estimation using $C * I")
    hline!(plt, [data[1][:true_logZ]], label="true value")
    #Plots.savefig(joinpath(plotdir, "$C*I.png"))
    display(plt)
    readline()
end

###### Cell ###### - create latex table of results
sort!(all_data, by=d->norm(d[:Σ₀]))

table_file = joinpath(target_root, "gaussian_covariance_table_1_2.tex")
open(table_file, "w") do io
    pretty_table(io,
                 hcat([ d[:failed_count]==0 ? round.(d[:estimation_rkhs], digits=2) : [round.(d[:estimation_rkhs], digits=2); NaN] for d in all_data[1:9] ]...),
                 hcat([ [d[:Σ₀], d[:update_method]] for d in [d for d in all_data[1:9]] ]...),
                 backend=:latex
                )
end

table_file = joinpath(target_root, "gaussian_covariance_table_2_2.tex")
open(table_file, "w") do io
    pretty_table(io,
                 hcat([ d[:failed_count]==0 ? round.(d[:estimation_rkhs], digits=2) : [round.(d[:estimation_rkhs], digits=2); NaN] for d in all_data[10:end] ]...),
                 hcat([ [d[:Σ₀], d[:update_method]] for d in [d for d in all_data[10:end]] ]...),
                 backend=:latex
                )
end

###### Cell ###### - load data # # Grid search over methods and some parameters
all_data = [BSON.load(n) for n in readdir(datadir("gaussian_to_gaussian", "method_compare"), join=true)];
all_data = [data for data in all_data if data[:failed_count]<10];
for d in all_data
    d[:svgd_hist] = convert(Array{MVHistory}, d[:svgd_hist])
end

###### Cell ###### - set up target dir for plots
target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/gauss/method_compare/")
mkpath(plotdir)

###### Cell ###### - boxplot for WNES with 10I initial covariance
WNES_path = joinpath(plotdir, "WNES")
mkpath(WNES_path)
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ 10*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]
                               ),
                          all_data)
    plt = make_boxplots(data, title="Σ=10*I, ϵ=$ϵ",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        integration_method=:trapz)
    display(plt)
    #    readline()
    #Plots.savefig(joinpath(WNES_path, "10I_stepsize=$ϵ.png"))
end
# For the larger covariance a small stepsize was best.
# The least variance in results was obtained with c₁ = c₂ = 0.1 and c₂ = 5, c₁ =-.1

## Cell - boxplot WNES for 0.1I initial covariance
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ .1*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]), all_data)
    plt = make_boxplots(data, title="Σ=0.1*I, ϵ=$ϵ",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        integration_method=:trapz)
    display(plt)
    #    readline()
    #Plots.savefig(joinpath(WNES_path, "0.1I_stepsize=$ϵ.png"))
end
# for small covariance only c₁=c₂=0.1 was any good

###### Cell ###### - boxplot WAG
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
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                            integration_method=:trapz)
        display(plt)
        #        readline()
        #Plots.savefig(joinpath(WAG_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

# α=7 was the best across all settings if there was any difference
# except large for with ϵ=0.05, which was the best overall
# in general all the estimates seem shit and mostly with high variance
# at small step sizes nothing happened

###### Cell ###### - boxplot RMSprop
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
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                            integration_method=:trapz)
        display(plt)
        #        readline()
        #Plots.savefig(joinpath(RMSprop_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

# for small covariance γ=0.8 was always good while γ=0.9 was only good with ϵ=0.005
# which large cov γ=0.8 was also consistently better (except for the case ϵ=0.005)
# where the again performed equally well but not very good overall
# (possible they just didn't converge)

###### Cell ###### - boxplot Adam
Adam_path = joinpath(plotdir, "Adam")
mkpath(Adam_path)
for C in [0.1, 10.]
    data = filter_by_dict( Dict(:update_method => [:scalar_Adam],
                                :Σ₀ => [ C*[1.0 0; 0 1] ],
                               ),
                          all_data)
    plt = make_boxplots(data, title="Σ=$(C)I",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        labels = ["ϵ = $(d[:step_size])" for d in data],
                        integration_method=:trapz)
    display(plt)
    # readline()
    #Plots.savefig(joinpath(Adam_path, "$(C)I.png"))
end

# The smallest step size gave the best results

## # Using trapezoid integration instead of upper Riemann sum

###### Cell ######
target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/gauss/integration_method_compare/")
mkpath(plotdir)

###### Cell ###### - load data
all_data = [BSON.load(n) for n in readdir(datadir("gaussian_to_gaussian", "method_compare"), join=true)];
all_data = [data for data in all_data if data[:failed_count]<10];
for d in all_data
    d[:svgd_hist] = convert(Array{MVHistory}, d[:svgd_hist])
end

###### Cell ###### - boxplot for WNES with 10I initial covariance
WNES_path = joinpath(plotdir, "WNES")
mkpath(WNES_path)
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ 10*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]
                               ),
                          all_data)
    plt = make_boxplots(data, title="Σ=10*I, ϵ=$ϵ",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        # integration_method=:trapz)
                       )
    display(plt)
    readline()
    #Plots.savefig(joinpath(WNES_path, "10I_stepsize=$ϵ.png"))
end
# For the larger covariance a small stepsize was best.
# The least variance in results was obtained with c₁ = c₂ = 0.1 and c₂ = 5, c₁ =-.1

## Cell - boxplot WNES for 0.1I initial covariance
for ϵ in [0.5, 0.05, 0.005]
    data = filter_by_dict( Dict(:update_method => [:naive_WNES],
                                :Σ₀ => [ .1*[1.0 0; 0 1] ],
                                :step_size => [ ϵ ]), all_data)
    plt = make_boxplots(data, title="Σ=0.1*I, ϵ=$ϵ",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        integration_method=:upper_sum)
    display(plt)
    readline()
    #Plots.savefig(joinpath(WNES_path, "0.1I_stepsize=$ϵ.png"))
end
# for small covariance only c₁=c₂=0.1 was any good

###### Cell ###### - boxplot WAG
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
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                            integration_method=:trapz)
        display(plt)
        readline()
        #Plots.savefig(joinpath(WAG_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

# α=7 was the best across all settings if there was any difference
# except large for with ϵ=0.05, which was the best overall
# in general all the estimates seem shit and mostly with high variance
# at small step sizes nothing happened

###### Cell ###### - boxplot RMSprop
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
                            ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                            integration_method=:trapz)
        display(plt)
        readline()
        #Plots.savefig(joinpath(RMSprop_path, "$(C)I_stepsize=$ϵ.png"))
    end
end

# for small covariance γ=0.8 was always good while γ=0.9 was only good with ϵ=0.005
# which large cov γ=0.8 was also consistently better (except for the case ϵ=0.005)
# where the again performed equally well but not very good overall
# (possible they just didn't converge)

###### Cell ###### - boxplot Adam
Adam_path = joinpath(plotdir, "Adam")
mkpath(Adam_path)
for C in [0.1, 10.]
    data = filter_by_dict( Dict(:update_method => [:scalar_Adam],
                                :Σ₀ => [ C*[1.0 0; 0 1] ],
                               ),
                          all_data)
    plt = make_boxplots(data, title="Σ=$(C)I",
                        ylims=(data[1][:true_logZ]-10,data[1][:true_logZ]+10),
                        labels = ["ϵ = $(d[:step_size])" for d in data],
                        integration_method=:trapz)
    display(plt)
    readline()
    #Plots.savefig(joinpath(Adam_path, "$(C)I.png"))
end

###### Cell ######
