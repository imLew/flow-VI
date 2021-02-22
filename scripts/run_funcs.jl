using DrWatson
using Distributions

using SVGD
using Utils
using Examples

function run_gauss_to_gauss(;problem_params, alg_params, n_runs, DIRNAME)
    svgd_results = []
    estimation_rkhs = []

    initial_dist = MvNormal(problem_params[:μ₀], problem_params[:Σ₀])
    target_dist = MvNormal(problem_params[:μₚ], problem_params[:Σₚ])

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        q, hist = SVGD.svgd_sample_from_known_distribution( initial_dist,
                            target_dist; alg_params=alg_params )

        H₀ = Distributions.entropy(initial_dist)
        EV = expectation_V( initial_dist, target_dist)

        est_logZ_rkhs = estimate_logZ(H₀, EV, KL_integral(hist)[end])

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
    end

    true_logZ = logZ(target_dist)

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    return tagsave(datadir(DIRNAME, file_prefix * ".bson"),
                merge(alg_params, problem_params, 
                      @dict(n_runs, true_logZ, estimation_rkhs, svgd_results)),
                safe=true, storepatch=true)
end

function run_log_regression(;problem_params, alg_params, n_runs, DIRNAME)
    LR = Examples.LogisticRegression
    svgd_results = []
    estimation_rkhs = []
    # estimation_unbiased = []
    # estimation_stein_discrep = []

    # dataset with labels
    D = LR.generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
                                                  n₁=problem_params[:n₁],
                                                  μ₀=problem_params[:μ₀],
                                                  μ₁=problem_params[:μ₁], 
                                                  Σ₀=problem_params[:Σ₀],
                                                  Σ₁=problem_params[:Σ₁],
                                                  n_dim=problem_params[:n_dim], 
                                                 )

    initial_dist = MvNormal(problem_params[:μ_initial], 
                            problem_params[:Σ_initial])

    grad_logp(w) = vec( - inv(problem_params[:Σ_initial])
                        * ( w-problem_params[:μ_initial] ) 
                        + LR.logistic_grad_logp(D, w)
                      )

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        q = rand(initial_dist, alg_params[:n_particles])
        q, hist = SVGD.svgd_fit(q=q, grad_logp=grad_logp; alg_params...)
        H₀ = Distributions.entropy(initial_dist)
        EV = ( SVGD.numerical_expectation( initial_dist, 
                                      w -> LR.logistic_log_likelihood(D,w) )
               + SVGD.expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
              )

        est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist)[end])
        est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :UKSB)[end])
        est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :KSD)[end])

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
        # don't forget to add these to the dict for saving below when using them again
        # push!(estimation_unbiased, est_logZ_unbiased)
        # push!(estimation_stein_discrep,est_logZ_stein_discrep)
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    return tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict(n_runs, estimation_rkhs, svgd_results)),
            safe=true, storepatch = false)
end
