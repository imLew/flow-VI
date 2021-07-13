using DrWatson
@quickactivate
using LinearAlgebra

using Utils

ALG_PARAMS = Dict(
    :step_size =>  0.001,
    :n_iter =>  20000,
    :n_particles =>  50,
    :update_method => [:WAG, :WNES, :forward_euler],
    :α => @onlyif(:update_method==:WAG, [3.01, 5, 10]),
    :c₁ => @onlyif(:update_method==:WNES, [1, 5, 10]),
    :c₂ => @onlyif(:update_method==:WNES, [1, 5, 10]),
    :kernel_cb => median_trick_cb!,
    :n_runs => 10,
    :random_seed => 0,
)

PROBLEM_PARAMS = Dict(
    :problem_type => :gauss_mixture_sampling,
    :n_dim => 2,
    :μ_initial => [ [0., 0.] ],
    :Σ_initial => [ [1. 0; 0 1.], ],
    :μₚ => [ [[8., -2.], [1., -6.], [4., -6.], [2., -3.]] ],
    :Σₚ => [ [I(2), I(2), I(2), I(2), I(2)] ],
    )

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, "WAG_WNes")
