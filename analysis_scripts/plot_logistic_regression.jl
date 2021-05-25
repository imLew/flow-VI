using Plots
using AdvancedHMC
using DrWatson
using ValueHistories
using Distributions
using LinearAlgebra
using BSON
using KernelFunctions
using PDMats
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples
LogReg = LogisticRegression

plotdir = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/bayesian_logistic_regression/"

all_data = [BSON.load(n) for n in readdir(datadir("bayesian_logistic_regression", "MAPvLaplacevNormal"), join=true) if endswith(n, ".bson")]
all_data = [data for data in all_data if data[:failed_count]<10]
for d in all_data
    d[:Σ_initial] = PDMat(Symmetric(d[:Σ_initial]))
    d[:Σ_prior] = PDMat(Symmetric(d[:Σ_prior]))
    d[:svgd_hist] = convert(Array{MVHistory}, d[:svgd_hist])
end

# savedir = plotdir * "forward_euler"
# data = filter_by_dict(Dict(:update_method => [:forward_euler]), all_data)
data = filter_by_dict(Dict(:Laplace_start => [true]), all_data)
for d in data
    readline()
    plt = plot_convergence(d)
    # plt = plot_classes(d, size=(300,220))
    # plot_prediction!(plt, d)
    # Plots.savefig(joinpath(plotdir, "classification_"*get_savename(d))*".png")
    display(plt)
end

for d in data
    plt = plot_integration(d, size=(300,200), ylims=(-30,Inf))
    display(plt)
    # Plots.savefig(joinpath(plotdir, "integration_"*get_savename(d))*".png")
    readline()
end

data = filter_by_dict(Dict(:update_method => [:naive_WAG],
                           :Laplace_start => [false]),
                      all_data)
for d in data
    readline()
    plt = plot_integration(d, size=(300,200), ylims=(-Inf,Inf),)
                           # title="c₁ = $(d[:c₁]), c₂ = $(d[:c₂])")
    display(plt)
    # Plots.savefig(joinpath(plotdir, "integration_"*get_savename(d))*".png")
end
