using DrWatson
using KernelFunctions
using BSON
using Distributions
using LinearAlgebra
using Plots

using Utils
using SVGD
using Examples
const LogReg = Examples.LogisticRegression

include("run_funcs.jl")

DIRNAME = "bayesian_logistic_regression"

alg_params = Dict(
    :update_method => [ :forward_euler ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :kernel_cb => [ median_trick_cb! ],
    :step_size => [ 0.001 ],
    :n_iter => [ 1000 ],
    :n_particles => [ 20 ],
    :n_runs => 1,
    )

problem_params = Dict(
    :problem_type => [ :bayesian_logistic_regression ],
    :MAP_start => [ false ],
    :MLE_start => [ false ],
    :Laplace_start => [ true ],
    :n_dim => [ 2 ],
    :n₀ => [ 50 ],
    :n₁ => [ 50 ],
    :μ₀ => [ [0., 0] ],
    :μ₁ => [ [4., 3] ],
    :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
    :Σ₁ => [ [.5 0.1; 0.1 .2] ],
    :μ_initial => [ [1., 1, 1] ],
    :Σ_initial => [ I(3) ],
    )

ap = dict_list(alg_params)[1]
pp = dict_list(problem_params)[1]
data = run_log_regression(pp, ap)

plt = plot_classes(data[:sample_data])
plot_prediction!(plt,data)

initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
H₀ = Distributions.entropy(initial_dist)
EV = ( numerical_expectation( initial_dist, 
            w -> LogReg.log_likelihood(data[:sample_data],w) )
       + expectation_V(initial_dist, initial_dist) 
       + 0.5 * logdet(2π * data[:Σ_initial])
) 

est_logZ_rkhs = estimate_logZ.(H₀, EV, KL_integral(data[:svgd_hist][1]))

norm_plot = plot(data[:svgd_hist][1][:ϕ_norm], title = "φ norm", yaxis = :log)
int_plot = plot( estimate_logZ.(H₀, EV, KL_integral(data[:svgd_hist][1])) ,title = "log Z", label = "",)

###############################################################################
## compare results
hist = data[:svgd_hist][1]

norm_plot = plot(get(hist,:ϕ_norm)[2], title="||φ||", yaxis=:log);
step_plot = plot(get(hist,:step_sizes)[2], title="ϵ", yaxis=:log);
# step_plot = plot(hist[:step_sizes], title="ϵ", yaxis=:log);
# cov_diff = norm.(get(hist, :Σ)[2][2:end] - get(hist, :Σ)[2][1:end-1])
# cov_plot = plot(cov_diff, title="||ΔΣ||", yaxis=:log);
int_plot = plot(title="log Z");
plot!(int_plot, estimate_logZ.([H₀], [EV], KL_integral(hist)), label="rkhs",);
# fit_plot = plot_results(plot(), q, problem_params);
plot(norm_plot, int_plot, step_plot, layout=@layout [n i; s])

@info "Value comparison" therm_logZ est_logZ_rkhs 
