using DrWatson
@quickactivate
using LinearAlgebra

using Utils

PROBLEM_PARAMS = Dict(
    :problem_type => [ :gauss_to_gauss ],
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [ 0.1*I(2), 10.0*I(2), 100.0*I(2), 100.0*I(2), ],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :dKL_estimator => [ :RKHS_norm ],
    :n_iter => [ 8000 ],
    :step_size => [ 0.01 ],
    :n_particles => [ 50 ],
    :update_method => [ :forward_euler ],
    :kernel_cb => [ median_trick_cb! ],
    :annealing_schedule => [ linear_annealing, hyperbolic_annealing,
                             cyclic_annealing ],
    :annealing_params => @onlyif(:annealing_schedule != linear_annealing,
                                 [ Dict(:p=>6, :C=>4),
                                   Dict(:p=>8, :C=>4),
                                   Dict(:p=>10, :C=>4) ]),
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS,
                    "gaussian_to_gaussian/annealing_second")
