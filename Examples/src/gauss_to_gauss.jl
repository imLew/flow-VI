using Distributions
using DrWatson

using SVGD
using Utils

export run_gauss_to_gauss

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
