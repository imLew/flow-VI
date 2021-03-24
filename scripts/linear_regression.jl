using DrWatson

if haskey(ENV, "JULIA_ENVIRONMENT")  # on the cluster
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else  # local
    @quickactivate
end

using BSON
# using Distributions
# using DataFrames
using LinearAlgebra
using KernelFunctions
using Plots
# using Distributions
using ValueHistories
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples
const LinReg = Examples.LinearRegression

include("run_funcs.jl")

DIRNAME = "linear_regression"

N_RUNS = 1

problem_params = Dict(
    :n_samples =>[ 20 ],
    :sample_range =>[ [-3, 3] ],
    :true_ϕ =>[ x -> [1, x, x^2] ],
    :true_w =>[ [2, -1, 0.2] ],
    :true_β =>[ 2 ],
    :ϕ =>[ x -> [1, x, x^2] ],
    :μ₀ =>[ zeros(3) ],
    :Σ₀ =>[ 0.1I(3) ],
    :MAP_start =>[ true ],
)

alg_params = Dict(
    :n_iter => [ 500 ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :step_size => [ 0.001 ],
    :n_particles => [ 50 ],
    :update_method => [:forward_euler],
    :α => @onlyif(:update_method == :naive_WAG, [3.1, 3.5, 5] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, .5,] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [3., 1] ),
    :kernel_cb => [ median_trick_cb! ],
    # :callback => [plot_cb]
)

function plot_results(plt, q, problem_params)
    x = range(problem_params[:sample_range]..., length=100)
    for w in eachcol(q)
        model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        plot!(plt,x, LinReg.y(model), alpha=0.3, color=:orange, legend=:none)
    end
    plot!(plt,x, 
            LinReg.y(LinReg.RegressionModel(problem_params[:ϕ], 
                              mean(q, dims=2), 
                              problem_params[:true_β])), 
        color=:red)
    plot!(plt,x, 
            LinReg.y(LinReg.RegressionModel(problem_params[:true_ϕ], 
                              problem_params[:true_w], 
                              problem_params[:true_β])), 
        color=:green)
    return plt
end

###############################################################################
## ThermoIntegration
## alg params 
nSamples = 3000
nSteps = 30
## alg
x = true_model.ϕ.(D.x)
prior = TuringDiagMvNormal(zeros(n_dim), ones(n_dim))
logprior(θ) = logpdf(prior, θ)
function loglikelihood(θ)
    (length(x)/2 * log(problem_params[:true_β]/2π) 
     - problem_params[:true_β]/2 * sum( (D.t .- dot.([θ], x)).^2 )
    )
end
θ_init = rand(n_dim)

alg = ThermoIntegration(nSamples = nSamples, nSteps=nSteps)
samplepower_posterior(x->loglikelihood(x) + logprior(x), n_dim, alg.nSamples)
therm_logZ = alg(logprior, loglikelihood, n_dim)


true_model = LR.RegressionModel(problem_params[:true_ϕ],
                             problem_params[:true_w], 
                             problem_params[:true_β])

D = LR.generate_samples(model=true_model, n_samples=problem_params[:n_samples],
                     sample_range=problem_params[:sample_range])
# scale = sum(extrema(D.t))
# problem_params[:true_w] ./= scale
# D.t ./= scale
###############################################################################
## SVGD integration

alg_params = Dict(
    :step_size => 0.00001,
    :n_iter => 2000,
    :n_particles => 100,
    :kernel_width => "median_trick",
)
initial_dist, q, hist = LR.fit_linear_regression(problem_params, alg_params, D)

H₀ = Distributions.entropy(initial_dist)
EV = ( 
LR.true_gauss_expectation(initial_dist,  
            LR.RegressionModel(problem_params[:ϕ], mean(initial_dist), 
                            problem_params[:true_β]), D)
        # num_expectation( initial_dist, 
        #         w -> log_likelihood(D, RegressionModel(problem_params[:ϕ], w, 
        #                                            problem_params[:true_β])) )
       + SVGD.expectation_V(initial_dist, initial_dist) 
       + 0.5 * logdet(2π * problem_params[:Σ_prior])
      )
est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, SVGD.KL_integral(hist)[end])
# est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV, KL_integral(hist, :dKL_unbiased)[end])
# est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV, KL_integral(hist, :dKL_stein_discrep)[end])

###############################################################################
## compare results

true_logZ = LR.regression_logZ(problem_params[:Σ_prior], problem_params[:true_β], 
                            problem_params[:true_ϕ], D.x)
