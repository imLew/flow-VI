using DrWatson
@quickactivate
using KernelFunctions
using LinearAlgebra

using SVGD
using Utils
using Examples

PROBLEM_PARAMS = Dict(
    :problem_type => [ :gauss_to_gauss ],
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [ 0.001*I(2), 0.01*I(2), 0.1*I(2), 10.0*I(2), 100.0*I(2), 1000.0*I(2) ],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :dKL_estimator => [ :RKHS_norm ],
    :n_iter => [2000],
    :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
    :step_size => [ 0.05, 0.005 ],
    :n_particles => [100],
    :update_method => [:forward_euler, :naive_WAG, :naive_WNES],
    :α => @onlyif(:update_method == :naive_WAG, [3.1] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, .5, @onlyif(:c₂==1.5, [1])] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [1, 1.5] ),
    :γ => @onlyif(:update_method == :scalar_RMS_prop, [.9, .8] ),
    :β₁ => @onlyif(:update_method == :scalar_Adam, [.9] ),
    :β₂ => @onlyif(:update_method == :scalar_Adam, [.999] ),
    :kernel_cb => [median_trick_cb!],
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, "gaussian_to_gaussian/WNESvWAGvEuler")
