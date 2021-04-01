using DrWatson
@quickactivate

using BSON
using LinearAlgebra
using KernelFunctions
using Plots
using Distributions
using ValueHistories
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples
const LinReg = Examples.LinearRegression

DIRNAME = "linear_regression"

PROBLEM_PARAMS = Dict(
    :problem_type => [ :linear_regression ],
    :n_samples =>[ 20 ],
    :sample_range =>[ [-3, 3] ],
    :true_ϕ =>[ x -> [1, x, x^2] ],
    :true_w =>[ [2, -1, 0.2] ],
    :true_β =>[ 2 ],
    :ϕ =>[ x -> [1, x, x^2] ],
    :μ_initial =>[ zeros(3) ],
    :Σ_initial =>[ I(3) ],
    :μ_prior =>[ zeros(3) ],
    :Σ_prior =>[ I(3) ],
    :MAP_start =>[ true ],
    :Laplace_start => [ true, false ],
    :Laplace_factor => @onlyif(:Laplace_start == true, [ 10., 0.1 ]),
    :therm_params => [Dict(
                          :nSamples => 2000,
                          :nSteps => 20
                         )],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :n_iter => [ 1000 ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :step_size => [ 0.01, 0.005 ],
    :n_particles => [ 50 ],
    :update_method => [ :scalar_RMS_prop, :naive_WNES, :scalar_Adam],
    :β₁ => [ 0.9 ],
    :β₂ => [ 0.999 ],
    :γ => [ 0.8 ] ,
    :c₁ => [ 0.1 ] ,
    :c₂ => [ 0.1 ] ,
    :kernel_cb => [ median_trick_cb! ],
    :dKL_estimator => [ :RKHS_norm ],
    :n_runs => [ 10 ],
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
