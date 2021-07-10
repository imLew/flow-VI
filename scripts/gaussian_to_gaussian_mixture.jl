using DrWatson
@quickactivate
using LinearAlgebra

using Utils

ALG_PARAMS = Dict(
    :step_size =>  0.01,
    :n_iter =>  20000,
    :n_particles =>  50,
    :dKL_estimator => :RKHS_norm,
    :update_method => :forward_euler,
    :kernel_cb => median_trick_cb!,
    :n_runs => 10,
    :random_seed => 0,
    :annealing_schedule => [linear_annealing, hyperbolic_annealing,
                            cyclic_annealing],
    :annealing_params => [
                      @onlyif(:annealing_schedule == linear_annealing,
                              Dict(:duration=>0.6)),
                      @onlyif(:annealing_schedule == hyperbolic_annealing,
                              Dict(:duration=>0.6, :p=>8), ),
                      @onlyif(:annealing_schedule == hyperbolic_annealing,
                              Dict(:duration=>0.8, :p=>8), ),
                      @onlyif(:annealing_schedule == hyperbolic_annealing,
                              Dict(:duration=>0.6, :p=>12)),
                      @onlyif(:annealing_schedule == hyperbolic_annealing,
                              Dict(:duration=>0.8, :p=>12)),
                      @onlyif(:annealing_schedule == cyclic_annealing,
                              Dict(:duration=>0.6, :p=>12, :C=>3)),
                         ],
)

PROBLEM_PARAMS = Dict(
    :problem_type => :gauss_mixture_sampling,
    :n_dim => 2,
    :μ_initial => [ [0., 0.] ],
    :Σ_initial => [ [1. 0; 0 1.], ],
    :μₚ => [ [[-8., -2.], [1., 6.], [2., -1.]] ],
    :Σₚ => [ [[1. 0.5; 0.5 1], [1.2 0.1; 0.1 1.2], I(2)] ],
    )

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS,
                    "gaussian_mixture_sampling/annealing")
