using BSON
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

all_data = [BSON.load(n) for n in readdir(datadir("gaussian_to_gaussian", "covariance"), join=true) ];

total = length(all_data)
all_data = [d for d in all_data if d[:failed_count] < 10];
not_failed = length(all_data)
@info "$not_failed out of $total had at least one complete run"
remaining_covs = Set([d[:Σ₀] for d in all_data])

target_root = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/"
plotdir = joinpath(target_root, "plots/covariance_all/")

for C in [0.001, 0.01, 0.1, 1, 10, 100]
    data = filter_by_dict( Dict(:Σ₀ => [ C*[1.0 0; 0 1] ]), all_data)

    plt = boxplot([d[:estimation_rkhs] for d in data],
            labels=reshape(["$(d[:update_method])" for d in data], 
                           (1,length(data))), 
            legend=:outerright; title="Log Z estimation using $C * I") 
    hline!(plt, [data[1][:true_logZ]], label="true value")
    Plots.savefig(joinpath(plotdir, "$C*I.png"))
end

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
