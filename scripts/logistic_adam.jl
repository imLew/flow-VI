using DrWatson
@quickactivate
using LinearAlgebra

using Utils

ALG_PARAMS = Dict(
    :update_method => [ :scalar_Adam, :forward_euler ],
    :β₁ => @onlyif(:update_method == :scalar_Adam, [0.0, 0.3, 0.5, 0.7, 0.9]),
    :β₂ => @onlyif(:update_method == :scalar_Adam, [0.0, 0.2, 0.5, 0.8, 0.999, 0.9999]),
    :kernel_cb => median_trick_cb!,
    :step_size => [ 0.001, @onlyif(:update_method == :forward_euler, 0.0001) ],
    :n_iter => [ 20000 ],
    :n_particles => 50,
    :n_runs => 10,
    :dKL_estimator => :RKHS_norm,
    :adam_unbiased => true,
    :adam_stepsize_method => :minimum,
    )

dict_list_count(ALG_PARAMS)

PROBLEM_PARAMS = Dict(
    :problem_type => :logistic_regression,
    :MAP_start => true,
    :n_dim => 2,
    :n₀ => 50,
    :n₁ => 50,
    :μ₀ => [ [0., 0] ],
    :μ₁ => [ [4., 3] ],
    :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
    :Σ₁ => [ [.5 0.1; 0.1 .2] ],
    :μ_prior => [ zeros(3) ],
    :Σ_prior => [ I(3) ],
    :μ_initial => [ [1., 1, 1] ],
    :Σ_initial => [ I(3) ],
    :random_seed => 0,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, "bayesian_logistic_regression/Adam")
