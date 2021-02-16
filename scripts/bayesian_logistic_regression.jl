using DrWatson
using KernelFunctions
using BSON

using Utils

include("run_funcs.jl")

DIRNAME = "bayesian_logistic_regression"

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
#             n_runs=N_RUNS,
#             DIRNAME=DIRNAME
#             )
# end
