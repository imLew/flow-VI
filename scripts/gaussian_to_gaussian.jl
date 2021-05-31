using DrWatson
@quickactivate
using LinearAlgebra

using Utils

PROBLEM_PARAMS = Dict(
    :problem_type => [ :gauss_to_gauss ],
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [ 10.0^i*I(2) for i in -3:0.2:3 ],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :dKL_estimator => [ :RKHS_norm ],
    :n_iter => [ 5000 ],
    :step_size => [ 0.01 ],
    :n_particles => [ 50 ],
    :update_method => [ :forward_euler ],
    :kernel_cb => [ median_trick_cb! ],
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS,
                    "gaussian_to_gaussian/cov_comparison",)
