using DrWatson 
@quickactivate
using KernelFunctions
using LinearAlgebra
using AdvancedHMC

using LoggingExtras
using Logging
function not_AdvancedHMC_message_filter(log)
    log._module != AdvancedHMC
end
global_logger(EarlyFilteredLogger(not_AdvancedHMC_message_filter, global_logger()))

using Utils
using SVGD

include("run_funcs.jl")

DIRNAME = "bayesian_logistic_regression/MAPvLaplacevNormal"

ALG_PARAMS = Dict(
    :update_method => [ :forward_euler ],
    :kernel => [ TransformedKernel(SqExponentialKernel(), ScaleTransform(1.)) ],
    :kernel_cb => [ median_trick_cb! ],
    :step_size => [ 0.001 ],
    :n_iter => [ 1000 ],
    :n_particles => [ 20 ],
    :n_runs => [ 2 ],
    )

PROBLEM_PARAMS = Dict(
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
    :random_seed => [ 0 ],
    :sample_data_file => [datadir("classification_samples", 
    "2dim_50:[0.0, 0.0]:[0.5 0.1; 0.1 0.2]_50:[0.0, 0.0]:[0.5 0.1; 0.1 0.2].bson")
                         ],
)

cmdline_run(ALG_PARAMS, PROBLEM_PARAMS, DIRNAME, run_log_regression)
