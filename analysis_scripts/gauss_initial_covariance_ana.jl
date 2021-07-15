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

target_plotdir = "gauss_cov"
plotdir = joinpath("/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/", target_plotdir)

mkpath(plotdir)

saveplot(f) = (savefig ∘ joinpath)(plotdir, f)
saveplot(args...) = (savefig ∘ joinpath)(plotdir, args...)

###### Cell ###### -
# PROBLEM_PARAMS = Dict(
#     :problem_type => :gauss_to_gauss,
#     :μ₀ => [[0., 0]],
#     :μₚ => [[0, 0]],
#     :Σₚ => [[1. 0; 0 1.]],
#     :Σ₀ => [10^c*I(2) for c in -6:0.5:6],
#     :random_seed => 0,
# )

# ALG_PARAMS = Dict(
#     :dKL_estimator => :RKHS_norm,
#     :n_iter => 10000,
#     :step_size => 0.01,
#     :n_particles => 50,
#     :update_method => :forward_euler,
#     :kernel_cb => median_trick_cb!,
#     :n_runs => 10,
# )

all_data = load_data( "gaussian_to_gaussian/initial_cov_comparison" );
sort!(all_data, by=d->norm(d[:Σ₀]));

###### Cell ###### -
plot_integration(all_data[end])

###### Cell ###### - relative performance vs log scaling
covs = [10.0^i for i in -6:0.5:6]
rel_err(d) = abs.(d[:estimated_logZ].-d[:true_logZ])./d[:true_logZ]

sort!(all_data, by=d->norm(d[:Σ₀]))
errs = [rel_err(d) for d in all_data]

scatter(log.(covs), mean.(errs), yerr=std.(errs), ylabel="Rel. Error Log Z",
        xlabel="Log c", legend=:none)
# hline!([0])
saveplot("logCov_full.png")

# scatter(covs, mean.(errs), yerr=log.(std.(errs)), ylabel="Rel. Error Log Z",
#         xlabel="Log c", legend=:none)
# hline!([0])
# saveplot("Cov.png")

###### Cell ###### - relative performance vs log scaling, truncated
# max e value = 25
s, e = 9, 25
covs = [10.0^i for i in -6:0.5:6][s:e]
errs = [rel_err(d) for d in all_data][s:e]

scatter(log10.(covs), log10.(mean.(errs)), yerr=log10.(std.(errs)), xticks=vec(-2:0.5:6),
        ylabel="Log₁₀(Rel. Error Log Z)", xlabel="Log₁₀ c", legend=:none)
saveplot("logCov.png")

###### Cell ###### - difference between target and start vs scaling
s, e = 13, 25
covs = [10.0^i for i in -6:0.5:6][s:e]
errs = [rel_err(d) for d in all_data][s:e]

KL(c) = 3/2*log(exp(c-1)/abs(c))
KL(c) = 3/2*(c-1-log(abs(c)))
# KLs = [KL(c) for c in covs];

plot(KL.(covs), mean.(errs), yerr=(std.(errs)),
     xlabel="KL(q₀||p)", ylabel="Rel. Error Log Z", label=:none)
saveplot("relerr_vs_KL.png")

###### Cell ###### -

###### Cell ###### -

###### Cell ###### -
