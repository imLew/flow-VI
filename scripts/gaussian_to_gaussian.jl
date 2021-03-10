#!/usr/bin/env julia
#$ -binding linear:16 # request cpus 
#$ -N gauss_to_gauss
#$ -q all.q 
#$ -cwd 
#$ -V 
#$ -t 1-16
### Run SVGD integration of KL divergence on the problem of smapling from
### a Gaussian 
########
### command line arguments:
### make-dicts - create the parameter dicts with DrWatson
### run "file" - run the algorithm on the parameters given in "file"
### run-all - run the script on every file specified in "_research/tmp"
### make-and-run-all - do `make-dicts` followed by `run-all`
#### Before executing `run-all` of `make-and-run-all` on the cluster the number
#### of tasks on line 7 ("#$ -t 1-N#Experiments) must be changed
using DrWatson

if haskey(ENV, "JULIA_ENVIRONMENT")  # on the cluster
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else  # local
    @quickactivate
end

using KernelFunctions
using Plots
using Distributions
using LinearAlgebra
using ValueHistories
using BSON
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples

PROBLEM_PARAMS = Dict(
    :problem_type => [ :gauss_to_gauss ],
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [10.0^i .* [1. 0; 0 1] for i in -3:1:3],
    :random_seed => [ 0 ],
)

# plt = plot()
# function plot_cb(;kwargs...)
#     @unpack q, i = kwargs
#     α = get(kwargs, :α, 0)
#     c₁ = get(kwargs, :c₁, 0)
#     c₂ = get(kwargs, :c₂, 0)
#     initial_dist = MvNormal(problem_params[:μ₀][1], problem_params[:Σ₀][1])
#     target_dist = MvNormal(problem_params[:μₚ][1], problem_params[:Σₚ][1])
#     title = α!=0 ? "WAG $α" : "WNES c₁=$c₁ c₂=$c₂"
#     display(plot_2D_results(initial_dist, target_dist, q, title=title))
# end

ALG_PARAMS = Dict(
    :n_iter => [1000],
    :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
    :step_size => [0.05],
    :n_particles => [50],
    :update_method => [:forward_euler, :naive_WAG, :naive_WNES],
    :α => @onlyif(:update_method == :naive_WAG, [3.1] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [.3] ),
    :kernel_cb => [median_trick_cb!],
    # :callback => [plot_cb],
    :n_runs => 10,
)

cmdline_run(ALG_PARAMS, PROBLEM_PARAMS, "gaussian_to_gaussian/covariance/", run)
