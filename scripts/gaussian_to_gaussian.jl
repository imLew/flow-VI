using DrWatson
@quickactivate
using LinearAlgebra

using Utils

PROBLEM_PARAMS = Dict(
    :problem_type => [ :gauss_to_gauss ],
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [ 0.1*I(2), 10.0*I(2), ],
    # :Σ₀ => [ 0.001*I(2), 0.01*I(2), 0.1*I(2), 10.0*I(2), 100.0*I(2), 1000.0*I(2), ],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :dKL_estimator => [ :RKHS_norm ],
    :n_iter => [ 3000, 4000 ],
    :step_size => [ 0.05 ],
    :n_particles => [ 100, 200 ],
    :update_method => [ :forward_euler ],
    :kernel_cb => [ median_trick_cb! ],
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, "gaussian_to_gaussian/initial_grid_follow_up")
