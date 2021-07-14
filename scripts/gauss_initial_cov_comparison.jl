using DrWatson
@quickactivate
using LinearAlgebra

using Utils

PROBLEM_PARAMS = Dict(
    :problem_type => :gauss_to_gauss,
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [10^c*I(2) for c in -6:0.5:6],
    :random_seed => 0,
)

ALG_PARAMS = Dict(
    :dKL_estimator => :RKHS_norm,
    :n_iter => 10000,
    :step_size => 0.01,
    :n_particles => 50,
    :update_method => :forward_euler,
    :kernel_cb => median_trick_cb!,
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS,
                    "gaussian_to_gaussian/initial_cov_comparison")
