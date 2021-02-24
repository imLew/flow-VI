using DrWatson
using Distributions
using Optim

using SVGD
using Utils
using Examples
const LinReg = Examples.LinearRegression

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

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
                merge(alg_params, problem_params, 
                      @dict(n_runs, true_logZ, estimation_rkhs, svgd_results)),
                safe=true, storepatch=true)
end

function fit_linear_regression(problem_params, alg_params, D::LinReg.RegressionData)
    function logp(w)
        model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        LinReg.log_likelihood(D, model) + logpdf(MvNormal(problem_params[:μ₀], problem_params[:Σ₀]), w)
    end  
    function grad_logp(w) 
        model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        (LinReg.grad_log_likelihood(D, model) 
         .- inv(problem_params[:Σ₀]) * (w-problem_params[:μ₀])
        )
    end
    grad_logp!(g, w) = g .= grad_logp(w)

    # use eithe prior as initial distribution of change initial mean to MAP
    problem_params[:μ₀] = if problem_params[:MAP_start]
        Optim.maximizer(Optim.maximize(logp, grad_logp!, problem_params[:μ₀], LBFGS()))
        # posterior_mean(problem_params[:ϕ], problem_params[:true_β], D, 
        #                problem_params[:μ₀], problem_params[:Σ₀])
    else
        problem_params[:μ₀]
    end

    initial_dist = MvNormal(problem_params[:μ₀], problem_params[:Σ₀])
    q = rand(initial_dist, alg_params[:n_particles])

    q, hist = SVGD.svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_linear_regression(problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_rkhs = []

    true_model = LinReg.RegressionModel(problem_params[:true_ϕ],
                                 problem_params[:true_w], 
                                 problem_params[:true_β])
    # dataset with labels
    sample_data = LinReg.generate_samples(model=true_model, 
                         n_samples=problem_params[:n_samples],
                         sample_range=problem_params[:sample_range]
                        )

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, q, hist = fit_linear_regression(problem_params, 
                                                      alg_params, sample_data)
        H₀ = Distributions.entropy(initial_dist)
        EV = ( num_expectation( 
                    initial_dist, 
                    w -> LinReg.log_likelihood(sample_data, 
                            LinReg.RegressionModel(problem_params[:ϕ], w, 
                                            problem_params[:true_β])) 
               )
               + expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ₀]) )
              )
        est_logZ_rkhs = estimate_logZ(H₀, EV, KL_integral(hist)[end])

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
    end

    true_logZ = LinReg.regression_logZ(problem_params[:Σ₀], true_model.β,
                                       true_model.ϕ, sample_data.x)

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict(n_runs, true_logZ, estimation_rkhs, svgd_results, sample_data) ),
            safe=true, storepatch = false)
end

# function run_log_regression(;problem_params, alg_params, n_runs, DIRNAME)
#     LinReg = Examples.LogisticRegression
#     svgd_results = []
#     estimation_rkhs = []
#     # estimation_unbiased = []
#     # estimation_stein_discrep = []

#     # dataset with labels
#     D = LinReg.generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
#                                                   n₁=problem_params[:n₁],
#                                                   μ₀=problem_params[:μ₀],
#                                                   μ₁=problem_params[:μ₁], 
#                                                   Σ₀=problem_params[:Σ₀],
#                                                   Σ₁=problem_params[:Σ₁],
#                                                   n_dim=problem_params[:n_dim], 
#                                                  )

#     initial_dist = MvNormal(problem_params[:μ_initial], 
#                             problem_params[:Σ_initial])

#     grad_logp(w) = vec( - inv(problem_params[:Σ_initial])
#                         * ( w-problem_params[:μ_initial] ) 
#                         + LinReg.logistic_grad_logp(D, w)
#                       )

#     for i in 1:n_runs
#         @info "Run $i/$(n_runs)"
#         q = rand(initial_dist, alg_params[:n_particles])
#         q, hist = SVGD.svgd_fit(q=q, grad_logp=grad_logp; alg_params...)
#         H₀ = Distributions.entropy(initial_dist)
#         EV = ( SVGD.numerical_expectation( initial_dist, 
#                                       w -> LinReg.logistic_log_likelihood(D,w) )
#                + SVGD.expectation_V(initial_dist, initial_dist) 
#                + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
#               )

#         est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist)[end])
#         est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :UKSB)[end])
#         est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist, :KSD)[end])

#         push!(svgd_results, (hist, q))
#         push!(estimation_rkhs, est_logZ_rkhs) 
#         # don't forget to add these to the dict for saving below when using them again
#         # push!(estimation_unbiased, est_logZ_unbiased)
#         # push!(estimation_stein_discrep,est_logZ_stein_discrep)
#     end

#     file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

#     return tagsave(datadir(DIRNAME, file_prefix * ".bson"),
#             merge(alg_params, problem_params, 
#                 @dict(n_runs, estimation_rkhs, svgd_results)),
#             safe=true, storepatch = false)
# end
