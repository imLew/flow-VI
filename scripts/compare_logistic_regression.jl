include("../scripts/bayesian_logistic_regression.jl")
include("../src/therm_int.jl")
include("../src/SVGD.jl")

include("scripts/bayesian_logistic_regression.jl")
include("src/therm_int.jl")

# set up
problem_params = Dict(
    :n_dim => 2,
    :n₀ => 100,
    :n₁ => 100,
    :μ₀ => [0., 0],
    :μ₁ => [5, 1],
    :Σ₀ => [1. 0.5; 0.5 1], 
    :Σ₁ => [5 0.1; 0.1 2],
    :μ_initial => [0., 0, 0],
    :Σ_initial => [1. 0 0; 0 1 0; 0 0 1.],
    )

D = generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
    n₁=problem_params[:n₁], μ₀=problem_params[:μ₀], μ₁=problem_params[:μ₁], 
    Σ₀=problem_params[:Σ₀], Σ₁=problem_params[:Σ₁], 
    n_dim=problem_params[:n_dim]
)

###############################################################################
## ThermoIntegration
## alg params 
nSamples = 3000
nSteps = 30
## alg

n_dim = 3
prior = TuringDiagMvNormal(zeros(n_dim), ones(n_dim))
logprior(θ) = logpdf(prior, θ)
loglikelihood(θ) = logistic_log_likelihood(D, θ)
θ_init = rand(n_dim)

alg = ThermoIntegration(nSamples = nSamples, nSteps=nSteps)
samplepower_posterior(x->loglikelihood(x) + logprior(x), n_dim, alg.nSamples)
therm_logZ = alg(logprior, loglikelihood, n_dim)

###############################################################################
## SVGD integration

alg_params = Dict(
    :step_size_cb => (s, i) -> SVGD.geometric_step_size_cb(s, i, 1.005, 500),
    :step_size => 0.00001,
    :n_iter => 2000,
    :n_particles => 20,
    :kernel => TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)),
    :kernel_cb => SVGD.median_trick_cb,
)
initial_dist, q, hist = fit_logistic_regression(problem_params, alg_params, D)

H₀ = Distributions.entropy(initial_dist)
EV = ( SVGD.numerical_expectation( initial_dist, 
    w -> logistic_log_likelihood(D,w) )
    + SVGD.expectation_V(initial_dist, initial_dist) 
    + 0.5 * logdet(2π * problem_params[:Σ_initial])
) 
est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist)[end])
est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :dKL_unbiased)[end])
est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :dKL_stein_discrep)[end])

file_prefix = savename( merge(problem_params, alg_params) )

tagsave(datadir(DIRNAME, file_prefix * ".bson"),
        merge(alg_params, problem_params, 
        @dict(est_logZ_unbiased, est_logZ_stein_discrep, est_logZ_rkhs, hist, q)),
        safe=true, storepatch = false)
        
###############################################################################
## compare results
using Plots

norm_plot = plot(get(hist,:ϕ_norm)[2], title="||φ||", yaxis=:log);
step_plot = plot(get(hist,:step_sizes)[2], title="ϵ", yaxis=:log);
# step_plot = plot(hist[:step_sizes], title="ϵ", yaxis=:log);
cov_diff = norm.(get(hist, :Σ)[2][2:end] - get(hist, :Σ)[2][1:end-1])
cov_plot = plot(cov_diff, title="||ΔΣ||", yaxis=:log);
int_plot = plot(title="log Z");
plot!(int_plot, SVGD.estimate_logZ.([H₀], [EV], SVGD.KL_integral(hist)),
                label="rkhs",);
plot!(int_plot, SVGD.estimate_logZ.([H₀], [EV], SVGD.KL_integral(hist, :dKL_stein_discrep)),
                label="discrep",);
plot!(int_plot, SVGD.estimate_logZ.([H₀], [EV], SVGD.KL_integral(hist, :dKL_unbiased)),
                label="unbiased",);
# fit_plot = plot_results(plot(), q, problem_params);
plot(norm_plot, int_plot, step_plot, cov_plot, layout=@layout [n i; s c])

@info "Value comparison" therm_logZ est_logZ_rkhs est_logZ_stein_discrep est_logZ_unbiased
