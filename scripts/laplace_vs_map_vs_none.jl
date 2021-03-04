using DrWatson 
@quickactivate
using KernelFunctions
using LinearAlgebra

using Utils
using SVGD

include("run_funcs.jl")

DIRNAME = "bayesian_logistic_regression/MAPvLaplacevNormal"

alg_params = Dict(
    :update_method => [ :forward_euler ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :kernel_cb => [ median_trick_cb! ],
    :step_size => [ 0.001 ],
    :n_iter => [ 1000 ],
    :n_particles => [ 20 ],
    :n_runs => [ 10 ],
    )

problem_params = Dict(
    :problem_type => [ :bayesian_logistic_regression ],
    :MAP_start => [ false, true ],
    :Laplace_start => [false, @onlyif(:MAP_start == true,  true )],
    :n_dim => [ 2 ],
    :n₀ => [ 50 ],
    :n₁ => [ 50 ],
    :μ₀ => [ [0., 0] ],
    :μ₁ => [ [4., 3] ],
    :Σ₀ => [ [0.5 0.1; 0.1 0.2] ],
    :Σ₁ => [ [.5 0.1; 0.1 .2] ],
    :μ_initial => [ [1., 1, 1] ],
    :Σ_initial => [ I(3) ],
    :therm_params => [Dict(
                          :nSamples => 3000,
                          :nSteps => 30
                         )],
    )

cmdline_run(alg_params, problem_params, DIRNAME, run_log_regression)
