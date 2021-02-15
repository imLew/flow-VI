using DrWatson
using KernelFunctions
using BSON
using LinearAlgebra
using Distributions

using SVGD
using Examples; const LR = Examples.LogisticRegression
using Utils

global DIRNAME = "bayesian_logistic_regression"

function fit_logistic_regression(problem_params, alg_params, D) 
    initial_dist = MvNormal(problem_params[:μ_initial],
                            problem_params[:Σ_initial])
    q = rand(initial_dist, alg_params[:n_particles])
    grad_logp(w) = vec( - inv(problem_params[:Σ_initial])
                     * ( w-problem_params[:μ_initial] ) 
                     + LR.logistic_grad_logp(D, w)
                    )

    q, hist = SVGD.svgd_fit(q=q, grad_logp=grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_log_regression(;problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_rkhs = []
    estimation_unbiased = []
    estimation_stein_discrep = []

    # dataset with labels
    D = LR.generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
                                              n₁=problem_params[:n₁],
                                              μ₀=problem_params[:μ₀],
                                              μ₁=problem_params[:μ₁], 
                                              Σ₀=problem_params[:Σ₀],
                                              Σ₁=problem_params[:Σ₁],
                                              n_dim=problem_params[:n_dim], 
                                             )

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, q, hist = fit_logistic_regression(problem_params, 
                                                        alg_params, D)
        H₀ = Distributions.entropy(initial_dist)
        EV = ( SVGD.numerical_expectation( initial_dist, 
                                      w -> LR.logistic_log_likelihood(D,w) )
               + SVGD.expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
              )

        est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist)[end])
        est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :dKL_unbiased)[end])
        est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :dKL_stein_discrep)[end])

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
        push!(estimation_unbiased, est_logZ_unbiased)
        push!(estimation_stein_discrep,est_logZ_stein_discrep)
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict(n_runs, estimation_unbiased, 
                        estimation_stein_discrep,
                        estimation_rkhs, svgd_results)),
            safe=true, storepatch = false)
end

alg_params = Dict(
    :kernel => TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)),
    :kernel_cb => median_trick_cb,
    :step_size => 0.05,
    :n_iter => 1000,
    :n_particles => 100,
    )

problem_params = Dict(
    :n_dim => 2,
    :n₀ => 100,
    :n₁ => 100,
    :μ₀ => [0., 0],
    :μ₁ => [5, 1],
    :Σ₀ => [.5 0.1; 0.1 0.2],
    :Σ₁ => [5 0.1; 0.1 2],
    :μ_initial => [0., 0, 0],
    :Σ_initial => [1. 0 0; 0 1 0; 0 0 1.],
    )

### Experiments - logistic regression on 2D vectors
ALG_PARAMS = Dict(
    :step_size => [0.05, 0.01, 0.005 ],
    :n_iter => [ 50, 100 ],
    :n_particles => [ 50, 100],
    :norm_method => "RKHS_norm",
    :kernel => TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)),
    :kernel_cb => median_trick_cb
    )

PROBLEM_PARAMS = Dict(
    :n_dim => 2,
    :n₀ => 100,
    :n₁ => 100,
    :μ₀ => [0., 0],
    :μ₁ => [5, 1],
    :Σ₀ => [.5 0.1; 0.1 0.2],
    :Σ₁ => [5 0.1; 0.1 2],
    :μ_initial => [ [0., 0, 0] ],
    :Σ_initial => [ [9. 0.5 1; 0.5 8 2;1 2 1.],  [1. 0 0; 0 1 0; 0 0 1.]  ],
    )

N_RUNS = 1

# n_alg = dict_list_count(ALG_PARAMS)
# n_prob = dict_list_count(PROBLEM_PARAMS)
# @info "$(n_alg*n_prob) total experiments"
# for (i, pp) ∈ enumerate(dict_list(PROBLEM_PARAMS)), 
#         (j, ap) ∈ enumerate(dict_list(ALG_PARAMS))
#     @info "Experiment $((i-1)*n_alg + j) of $(n_alg*n_prob)"
#     @info "Sampling problem: $pp"
#     @info "Alg parameters: $ap"
#     @time run_log_regression(
#             problem_params=pp,
#             alg_params=ap,
#             n_runs=N_RUNS
#             )
# end
