using DrWatson
@quickactivate
using LinearAlgebra

using Utils


PROBLEM_PARAMS = Dict(
    :problem_type => :gauss_to_gauss,
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [ 0.1*I(2), 10.0*I(2), ],
    :random_seed => 0,
)

ALG_PARAMS = Dict(
    :dKL_estimator => :RKHS_norm,
    :n_iter => 10000,
    :step_size => [ 0.01, 0.001, 0.005 ],
    :n_particles => [50, 100, 200],
    :update_method => :forward_euler,
    :kernel_cb => median_trick_cb!,
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS,
                    "gaussian_to_gaussian/number_particles",)
