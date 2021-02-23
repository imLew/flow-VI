#!/usr/bin/env julia
#$ -binding linear:16 # request cpus 
#$ -N gauss_to_gauss
#$ -q all.q 
#$ -cwd 
#$ -V 
#$ -t 1-16
### Run SVGD integration of KL divergence on the problem of smapling from
### a Gaussian starting from a standard gaussian
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
using ValueHistories
using BSON
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using SVGD
using Utils
using Examples

include("run_funcs.jl")

DIRNAME = "gaussian_to_gaussian"

N_RUNS = 1

problem_params = Dict(
    :μ₀ => [[0., 0]],
    :μₚ => [[0, 0]],
    :Σₚ => [[1. 0; 0 1.]],
    :Σ₀ => [[2. 0.5; 0.5 2], [0.3 0; 0 0.3]],
)

plt = plot()
function plot_cb(;kwargs...)
    @unpack q, i = kwargs
    α = get(kwargs, :α, 0)
    c₁ = get(kwargs, :c₁, 0)
    c₂ = get(kwargs, :c₂, 0)
    initial_dist = MvNormal(problem_params[:μ₀][1], problem_params[:Σ₀][1])
    target_dist = MvNormal(problem_params[:μₚ][1], problem_params[:Σₚ][1])
    title = α!=0 ? "WAG $α" : "WNES c₁=$c₁ c₂=$c₂"
    display(plot_2D_results(initial_dist, target_dist, q, title=title))
end

alg_params = Dict(
    :n_iter => [1000],
    :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
    :step_size => [0.05],
    :n_particles => [20],
    :update_method => [:naive_WNES, :naive_WAG],
    :α => @onlyif(:update_method == :naive_WAG, [3.1, 3.5,  5] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, .5,] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [3., 1] ),
    :kernel_cb => [median_trick_cb!],
    :callback => [plot_cb]
)

runs = []

recent_runs = []
n_sets = dict_list_count(alg_params)*dict_list_count(problem_params)
for (i, ap) ∈ enumerate(dict_list(alg_params))
    for (j, pp) ∈ enumerate(dict_list(problem_params))
        # @show ap[:update_method]
        # if haskey(ap, :c₁) 
        #     @show (ap[:c₁], ap[:c₂]) 
        # end
        # if haskey(ap, :α) 
        #     @show ap[:α] 
        # end
        println("$(((i-1)*dict_list_count(problem_params)) + j) out of $n_sets")
        name = run_gauss_to_gauss(problem_params=pp, alg_params=ap, 
                                  n_runs=N_RUNS, DIRNAME=DIRNAME)
        push!(recent_runs, name)
        display(plot_convergence(name))
    end
    if readline() == "q"
        break
    end
end

push!(runs, recent_runs)

all_data = [BSON.load(n) for n in readdir("data/gaussian_to_gaussian", join=true)]

# cmdline_run(N_RUNS, alg_params, problem_params, run_g2g)
