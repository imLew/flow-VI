#!/usr/bin/env julia
#$ -binding linear:16 # request cpus 
#$ -N gauss_to_gauss
#$ -q all.q 
#$ -cwd 
#$ -V 
#$ -t 1-16
### Run SVGD integration of KL divergence on the problem of smapling from
### a Gaussian starting from a standard gaussian
########
### command line arguments:
### make-dicts - create the parameter dicts with DrWatson
### run "file" - run the algorithm on the parameters given in "file"
### run-all - run the script on every file specified in "_research/tmp"
### make-and-run-all - do `make-dicts` followed by `run-all`
#### Before executing `run-all` of `make-and-run-all` on the cluster the number
#### of tasks on line 17 ("#$ -t 1-N#Experiments) must be changed
using DrWatson

using SVGD
using Utils
using Distributions

if haskey(ENV, "JULIA_ENVIRONMENT")
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else
    @quickactivate
end

global DIRNAME = "gaussian_to_gaussian"

### local util functions
function gaussian_to_gaussian(;μ₀::Vector, μₚ::Vector, Σ₀, Σₚ, alg_params)
    initial_dist = MvNormal(μ₀, Σ₀)
    target_dist = MvNormal(μₚ, Σₚ)
    q, hist = SVGD.svgd_sample_from_known_distribution( initial_dist, target_dist; 
                                                 alg_params=alg_params )
    return initial_dist, target_dist, q, hist
end

function run_g2g(;problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_rkhs = []
    estimation_unbiased = []
    estimation_stein_discrep = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, target_dist, q, hist = gaussian_to_gaussian( 
            ;problem_params..., alg_params=alg_params)

        H₀ = Distributions.entropy(initial_dist)
        EV = expectation_V( initial_dist, target_dist)

        est_logZ_rkhs = estimate_logZ(H₀, EV, KL_integral(hist)[end])
        est_logZ_unbiased = estimate_logZ(H₀, EV, KL_integral(hist, :dKL_unbiased)[end])
        est_logZ_stein_discrep = estimate_logZ(H₀, EV, KL_integral(hist, :dKL_stein_discrep)[end])

        global true_logZ = logZ(target_dist)

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
        push!(estimation_unbiased, est_logZ_unbiased)
        push!(estimation_stein_discrep,est_logZ_stein_discrep)
               
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                  @dict(n_runs, true_logZ, estimation_unbiased, 
                        estimation_stein_discrep,
                        estimation_rkhs, svgd_results)),
            safe=true, storepatch=true
    )
end

global N_RUNS = 10

PROBLEM_PARAMS = Dict(
    :μ₀ => [[0., 0]],
    :μₚ => [[4,5]],
    :Σ₀ => [[1. 0; 0 1.]],
    :Σₚ => [[1. 0.5; 0.5 1]],
)

ALG_PARAMS = Dict(
    :n_iter => [10000],
    :step_size => [0.05],
    :n_particles => [50, 100, 200, 500, 1000],
)

cmdline_run(N_RUNS, ALG_PARAMS, PROBLEM_PARAMS, run_g2g)
