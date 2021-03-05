using DrWatson 
@quickactivate
using KernelFunctions
using LinearAlgebra
using AdvancedHMC

using LoggingExtras
using Logging
function not_HTTP_message_filter(log)
    log._module != AdvancedHMC
end
global_logger(EarlyFilteredLogger(not_HTTP_message_filter, global_logger()))

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
    :n_runs => [ 1 ],
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
    # :therm_params => [Dict(
    #                       :nSamples => 3000,
    #                       :nSteps => 30
    #                      )],
    :random_seed => [ 5 ],
    )

pp = dict_list(problem_params)[3]
ap = dict_list(alg_params)[1]

for i in 1:10
    run_log_regression(problem_params=pp, alg_params=ap, DIRNAME="", save=false)
end

# D = LogReg.generate_2class_samples_from_gaussian(n₀=pp[:n₀],
#                                                   n₁=pp[:n₁],
#                                                   μ₀=pp[:μ₀],
#                                                   μ₁=pp[:μ₁], 
#                                                   Σ₀=pp[:Σ₀],
#                                                   Σ₁=pp[:Σ₁],
#                                                  )
# wsave(datadir("classification_samples",
#               "2dim_$(pp[:n₀]):$(pp[:μ₀]):$(pp[:Σ₀])_$(pp[:n₀]):$(pp[:μ₀]):$(pp[:Σ₀]).bson"),
#       @dict D)

# cmdline_run(alg_params, problem_params, DIRNAME, run_log_regression)
