using DrWatson
@quickactivate

using BSON
using LinearAlgebra
using Plots
using Distributions
using ValueHistories
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples
LinReg = Examples.LinearRegression

DIRNAME = "linear_regression/WNES_grid_search"

PROBLEM_PARAMS = Dict(
    :problem_type => [ :linear_regression ],
    :n_samples =>[ 10 ],
    :sample_range =>[ [-3, 3] ],
    :true_ϕ =>[ x -> [1, x, x^2] ],
    :true_w =>[ [2, -1, 0.2] ],
    :true_β =>[ 2 ],
    :ϕ =>[ x -> [1, x, x^2] ],
    :μ_initial =>[ zeros(3) ],
    :Σ_initial =>[ [1. 0 0; 0 1 0; 0 0 1] ],
    :μ_prior =>[ zeros(3) ],
    :Σ_prior =>[ [1. 0 0; 0 1 0; 0 0 1] ],
    :MAP_start =>[ true ],
    :Laplace_start => [ true, false ],
    :Laplace_factor => @onlyif(:Laplace_start == true, [ 10., 0.1 ]),
    # :therm_params => [Dict
    #                       :nSamples => 2000,
    #                       :nSteps => 20
    #                      )],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :n_iter => [ 1000 ],
    :step_size => [ 0.010, 0.015 ],
    :n_particles => [ 50 ],
    :update_method => [ :naive_WNES],
    :β₁ => [ 0.9 ],
    :β₂ => [ 0.999 ],
    :γ => [ 0.8 ] ,
    :c₁ => [ 0.3, 0.5, 0.1 ] ,
    :c₂ => [ 0.3, 0.5, 0.1 ] ,
    :kernel_cb => [ median_trick_cb! ],
    :dKL_estimator => [ :RKHS_norm ],
    :n_runs => [ 10 ],
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
