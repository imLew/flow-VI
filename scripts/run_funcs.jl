using DrWatson
using Distributions
using Optim

using SVGD
using Utils
using Examples
const LinReg = Examples.LinearRegression
const LogReg = Examples.LogisticRegression

# function run(problem_params, alg_params, n_runs, DIRNAME)
#     if problem_params[:problem_type] == :bayesian_logistic_regression
#         if get(problem_params, :MAP_start, false) == true
#             function logp(w)
#                 model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
#                 LinReg.log_likelihood(D, model) + logpdf(MvNormal(problem_params[:μ₀], problem_params[:Σ₀]), w)
#             end  
#             function grad_logp(w) 
#                 model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β)]
#                 (LinReg.grad_log_likelihood(D, model) 
#                  .- inv(problem_params[:Σ₀]) * (w-problem_params[:μ₀])
#                 )
#             end
#             grad_logp!(g, w) = g .= grad_logp(w)

#             problem_params[:μ₀] = Optim.maximizer(Optim.maximize(logp, 
#                                                                  grad_logp!, 
#                                                                  problem_params[:μ₀], 
#                                                                  LBFGS())
#                                                  )
#         end
#     elseif problem_params[:problem_type] == :gauss_to_gauss
#     elseif problem_params[:problem_type] == :bayesian_linear_regression
#     end
# end

function run_gauss_to_gauss(;problem_params, alg_params, DIRNAME)
    svgd_results = []
    svgd_hist = []
    estimation_rkhs = []

    initial_dist = MvNormal(problem_params[:μ₀], problem_params[:Σ₀])
    target_dist = MvNormal(problem_params[:μₚ], problem_params[:Σₚ])

    for i in 1:alg_params[:n_runs]
        @info "Run $i/$(alg_params[:n_runs])"
        q, hist = SVGD.svgd_sample_from_known_distribution( initial_dist,
                            target_dist; alg_params=alg_params )

        H₀ = Distributions.entropy(initial_dist)
        EV = expectation_V( initial_dist, target_dist)

        est_logZ_rkhs = estimate_logZ(H₀, EV, KL_integral(hist)[end])

        push!(svgd_results, q)
        push!(svgd_hist, hist)
        push!(estimation_rkhs, est_logZ_rkhs) 
    end

    true_logZ = logZ(target_dist)

    file_prefix = savename( merge(problem_params, alg_params) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
                merge(alg_params, problem_params, 
                      @dict(true_logZ, estimation_rkhs, svgd_results,
                           svgd_hist)),
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
    if problem_params[:MAP_start]
        problem_params[:μ₀] = Optim.maximizer(
                                Optim.maximize(logp, grad_logp!, 
                                               problem_params[:μ₀], LBFGS())
                               )
        # posterior_mean(problem_params[:ϕ], problem_params[:true_β], D, 
        #                problem_params[:μ₀], problem_params[:Σ₀])
    end

    initial_dist = MvNormal(problem_params[:μ₀], problem_params[:Σ₀])
    q = rand(initial_dist, alg_params[:n_particles])

    q, hist = SVGD.svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_linear_regression(problem_params, alg_params)
    svgd_results = []
    svgd_hist = []
    estimation_rkhs = []

    true_model = LinReg.RegressionModel(problem_params[:true_ϕ],
                                 problem_params[:true_w], 
                                 problem_params[:true_β])
    # dataset with labels
    sample_data = LinReg.generate_samples(model=true_model, 
                         n_samples=problem_params[:n_samples],
                         sample_range=problem_params[:sample_range]
                        )

    for i in 1:alg_params[:n_runs]
        @info "Run $i/$(alg_params[:n_runs])"
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

        push!(svgd_results, q)
        push!(svgd_hist, hist)
        push!(estimation_rkhs, est_logZ_rkhs) 
    end

    true_logZ = LinReg.regression_logZ(problem_params[:Σ₀], true_model.β,
                                       true_model.ϕ, sample_data.x)

    file_prefix = savename( merge(problem_params, alg_params) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict(true_logZ, estimation_rkhs, svgd_results, 
                      svgd_hist, sample_data)),
            safe=true, storepatch = false)
end

function fit_logistic_regression(problem_params, alg_params, D) 
    function logp(w)
        ( LogReg.log_likelihood(D, w) 
            + logpdf(MvNormal(problem_params[:μ₀], problem_params[:Σ₀]), w)
        )
    end  

    function grad_logp(w) 
        ( LogReg.grad_log_likelihood(D, w)
         .- inv(problem_params[:Σ₀]) * (w-problem_params[:μ₀])
        )
    end
    grad_logp!(g, w) = g .= grad_logp(w)
    @info test

    # use eithe prior as initial distribution of change initial mean to MAP
    if problem_params[:MAP_start] || problem_params[:Laplace_start]
        problem_params[:μ_initial] = Optim.maximizer(
                                Optim.maximize(logp, grad_logp!, 
                                               problem_params[:μ₀], LBFGS())
                               )
        @info "MAP estimate is " problem_params[:μ_initial]
        # posterior_mean(problem_params[:ϕ], problem_params[:true_β], D, 
        #                problem_params[:μ₀], problem_params[:Σ₀])
    end
    if problem_params[:Laplace_start]
        y = LogReg.y(D, problem_params[:μ_initial][1])
        problem_params[:Σ_initial] = ( inv(problem_params[:Σ_initial]) 
                                      + D.z' * ( y.*(1 .- y) .* D.z )
                                      )
        @info "Using Laplace approximation with " problem_params[:Σ_initial]
    end

    initial_dist = MvNormal(problem_params[:μ_initial], problem_params[:Σ_initial])

    q = rand(initial_dist, alg_params[:n_particles])
    # grad_logp(w) = vec( - inv(problem_params[:Σ_initial])
    #                  * ( w-problem_params[:μ_initial] ) 
    #                  + LogReg.grad_log_likelihood(D, w)
    #                 )

    q, hist = svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_log_regression(problem_params, alg_params)
    LinReg = Examples.LogisticRegression
    svgd_hist = []
    svgd_results = []
    estimation_rkhs = []

    # dataset with labels
    D = sample_data = LogReg.generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
                                                  n₁=problem_params[:n₁],
                                                  μ₀=problem_params[:μ₀],
                                                  μ₁=problem_params[:μ₁], 
                                                  Σ₀=problem_params[:Σ₀],
                                                  Σ₁=problem_params[:Σ₁],
                                                 )

    function logp(w)
        ( LogReg.log_likelihood(D, w) 
            + logpdf(MvNormal(problem_params[:μ_initial], problem_params[:Σ_initial]), w)
        )
    end  
    function grad_logp(w) 
        vec( LogReg.grad_log_likelihood(D, w)
         .- inv(problem_params[:Σ_initial]) * (w-problem_params[:μ_initial])
        )
    end
    grad_logp!(g, w) = g .= grad_logp(w)

    if problem_params[:MAP_start] || problem_params[:Laplace_start]
        problem_params[:μ_initial] = Optim.maximizer(
                                Optim.maximize(logp, grad_logp!, 
                                               problem_params[:μ_initial], LBFGS())
                               )
        @info "MAP estimate is " problem_params[:μ_initial]
    end
    if problem_params[:Laplace_start]
        y = LogReg.y(D, problem_params[:μ_initial])
        problem_params[:Σ_initial] = inv( get_pdmat( inv(problem_params[:Σ_initial]) 
                                           .+ D.z' * ( y.*(1 .- y) .* D.z )
                                          ) )
        @info "Using Laplace approximation with " problem_params[:Σ_initial]
    end

    therm_logZ = if haskey(problem_params, :therm_params)
        therm_integration(problem_params, D; problem_params[:therm_params]...)
    else
        nothing
    end

    initial_dist = MvNormal(problem_params[:μ_initial], problem_params[:Σ_initial])
    H₀ = Distributions.entropy(initial_dist)
    EV = ( num_expectation( initial_dist, 
                                  w -> LogReg.log_likelihood(sample_data,w) )
           + expectation_V(initial_dist, initial_dist) 
           + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
          )
    isnothing(therm_logZ) ? nothing : @show therm_logZ
    for i in 1:alg_params[:n_runs]
        @info "Run $i/$(alg_params[:n_runs])"
        q = rand(initial_dist, alg_params[:n_particles])
        q, hist = svgd_fit(q, grad_logp; alg_params...)

        push!(svgd_results, q)
        push!(svgd_hist, hist)
        est_logZ = estimate_logZ(H₀, EV, KL_integral(hist)[end])
        push!(estimation_rkhs, est_logZ) 
        @show est_logZ
    end

    file_prefix = savename( merge(problem_params, alg_params) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict(estimation_rkhs, svgd_results, svgd_hist,
                      sample_data, therm_logZ)),
            safe=true, storepatch = false)
end

function therm_integration(problem_params, D; nSamples=3000, nSteps=30)
    n_dim = problem_params[:n_dim] + 1
    # prior = TuringDiagMvNormal(zeros(n_dim), ones(n_dim))
    prior = MvNormal(zeros(n_dim), ones(n_dim))
    logprior(θ) = logpdf(prior, θ)
    loglikelihood(θ) = LogReg.log_likelihood(D, θ)
    θ_init = randn(n_dim)

    alg = ThermoIntegration(nSamples = nSamples, nSteps=nSteps)
    samplepower_posterior(x->loglikelihood(x) + logprior(x), n_dim, alg.nSamples)
    alg(logprior, loglikelihood, n_dim)  # log Z estimate
end
