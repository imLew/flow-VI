using DrWatson

if haskey(ENV, "JULIA_ENVIRONMENT")  # on the cluster
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else 
    @quickactivate
end

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
    :Laplace_start => [ true],
    :therm_params => [Dict(
                          :nSamples => 3000,
                          :nSteps => 30
                         )],
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :n_iter => [ 500 ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :step_size => [ 0.001 ],
    :n_particles => [ 50 ],
    :update_method => [:forward_euler],
    :α => @onlyif(:update_method == :naive_WAG, [3.1, 3.5, 5] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, .5,] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [3., 1] ),
    :kernel_cb => [ median_trick_cb! ],
    :dKL_estimator => [ :RKHS_norm ],
    :n_runs => [ 5 ],
    # :callback => [plot_cb]
)

pp = dict_list(PROBLEM_PARAMS)[1]
ap = dict_list(ALG_PARAMS)[1]

d = run_svgd(problem_params=pp, alg_params=ap, save=false)
plot_convergence(d)

true_model = LinReg.RegressionModel(pp[:true_ϕ],
                             pp[:true_w], 
                             pp[:true_β])
D = LinReg.generate_samples(model=true_model, 
                     n_samples=pp[:n_samples],
                     sample_range=pp[:sample_range]
                    )
