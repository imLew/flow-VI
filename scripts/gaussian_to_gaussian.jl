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
#### of tasks on line 17 ("#$ -t 1-N#Experiments) must be changed
using DrWatson
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

if haskey(ENV, "JULIA_ENVIRONMENT")
    quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")
else
    @quickactivate
end

DIRNAME = "gaussian_to_gaussian"

N_RUNS = 1

problem_params = Dict(
    :μ₀ => [[0., 0]],
    :μₚ => [[4,5]],
    :Σ₀ => [[1. 0; 0 1.]],
    :Σₚ => [[1. 0.5; 0.5 1]],
)

alg_params = Dict(
    :n_iter => [1000],
    :kernel => [TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))],
    :step_size => [0.05],
    :n_particles => [50],
    :update => ["vanilla", "naive_WAG"],
    :α => @onlyif(:update == "naive_WAG", [3.5, 4, 5, 7, 10] ),
    :kernel_cb => [median_trick_cb],
)

runs = []

recent_runs = []
n_sets = dict_list_count(alg_params)*dict_list_count(problem_params)
for (i, ap) ∈ enumerate(dict_list(alg_params))
    for (j, pp) ∈ enumerate(dict_list(problem_params))
        println("$(((i-1)*dict_list_count(problem_params)) + j) out of $n_sets")
        name = run_gauss_to_gauss(problem_params=pp,
                                  alg_params=ap, n_runs=N_RUNS,
                                  DIRNAME=DIRNAME)
        push!(recent_runs, name)
    end
end
push!(runs, recent_runs)

function plot_data(data; size=(275,275), legend=:bottomright, ylims=(-Inf,Inf), lw=3)
    initial_dist = MvNormal(data[1][:μ₀], data[1][:Σ₀])
    target_dist = MvNormal(data[1][:μₚ], data[1][:Σₚ])

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)

    int_plot = plot(xlabel="iterations", ylabel="log Z", legend=legend, size=size, lw=lw);

    for (i, d) in enumerate(data)
        dKL_hist = d[:svgd_results][1][1]
        est_logZ = estimate_logZ.([H₀], [EV], d[:step_size]*cumsum(get(dKL_hist, :dKL_rkhs)[2]));
        α = haskey(d, :α) ? "alpha = $(d[:α])" : ""
        plot!(int_plot, est_logZ, label="$(d[:update]) $α", color=colors[i+1], lw=lw);
    end
    hline!(int_plot, [true_logZ],ylims=ylims, color=colors[1], label="true value", lw=lw);
    return int_plot
end

function plot_all(data; size=(175,175), legend=:bottomright, ylims=(-Inf,Inf), lw=3)
    dKL_hist = data[:svgd_results][1][1]
    final_particles = data[:svgd_results][1][end]
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)
    int_plot = plot(xlabel="iterations", ylabel="log Z", legend=legend, lw=lw, ylims=(1, 3));
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_rkhs)[2]))
    plot!(int_plot, est_logZ, label="", color=colors[1]);
    hline!(int_plot, [true_logZ], labels="", color=colors[2], ls=:dash);
    dist_plot = plot_2D(initial_dist, target_dist, final_particles);
    if data[:n_iter] == 5000
        xticks=0:2500:5000
    elseif data[:n_iter] == 10000
        xticks=0:5000:10000
    elseif data[:n_iter] == 2000
        xticks=0:1000:2000
    end
    norm_plot = plot(data[:svgd_results][1][1][:ϕ_norm],ylims=(0,Inf),
                     markeralpha=0, label="", title="", xticks=xticks, color=colors[1],
                    xlabel="iterations", ylabel="||φ||");
    layout = @layout [ i ; n b]
    final_plot = plot(int_plot, norm_plot, dist_plot, layout=layout, legend=:bottomright, size=size);
end

all_data = [BSON.load(n) for n in readdir("data/gaussian_to_gaussian", join=true)]
plot_data(all_data, legend=:topleft)

# PROBLEM_PARAMS = Dict(
#     :μ₀ => [[0., 0]],
#     :μₚ => [[4,5]],
#     :Σ₀ => [[1. 0; 0 1.]],
#     :Σₚ => [[1. 0.5; 0.5 1]],
# )

# ALG_PARAMS = Dict(
#     :n_iter => [10000],
#     :step_size => [0.05],
#     :n_particles => [50, 100, 200, 500, 1000],
# )

# cmdline_run(N_RUNS, ALG_PARAMS, PROBLEM_PARAMS, run_g2g)
