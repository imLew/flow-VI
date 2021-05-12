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
    :Σ₀ => [ 0.1*I(2), 10.0*I(2), ],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :dKL_estimator => [ :RKHS_norm ],
    :n_iter => [1000, 2000],
    :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
    :step_size => [ 0.05, 0.001, 0.005 ],
    :n_particles => [50, 100],
    :update_method => [:forward_euler, :scalar_Adam, :scalar_adagrad,
                       @onlyif(:n_iter == 1000, [:scalar_RMS_prop])],
    :β₁ => @onlyif(:update_method==:scalar_Adam ,[ 0.9 ]),
    :β₂ => @onlyif(:update_method==:scalar_Adam ,[ 0.999 ]),
    :γ => @onlyif(:update_method==:scalar_RMS_prop ,[ 0.8, 0.9 ]),
    :kernel_cb => [median_trick_cb!],
    :n_runs => 10,
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, "gaussian_to_gaussian/gd_variants")
