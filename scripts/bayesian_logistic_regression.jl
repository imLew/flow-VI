using DrWatson
@quickactivate
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
using Examples

DIRNAME = "bayesian_logistic_regression/MAPvLaplacevNormal"

ALG_PARAMS = Dict(
    :update_method => [ :forward_euler ],
    :α => @onlyif(:update_method == :naive_WAG, [3, 4, 7, 10] ),
    :c₁ => @onlyif(:update_method == :naive_WNES, [.1, 1, 5] ),
    :c₂ => @onlyif(:update_method == :naive_WNES, [.1, 1, 5] ),
    :kernel_cb => [ median_trick_cb! ],
    :step_size => [ 0.001 ],
    :n_iter => [ 1000 ],
    :n_particles => [ 50 ],
    :n_runs => [ 10 ],
    :dKL_estimator => [ :RKHS_norm ],
    )

PROBLEM_PARAMS = Dict(
    :problem_type => [ :logistic_regression ],
    :MAP_start => [  true, false ],
    :Laplace_start => [ false,  true ],
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
    # :sample_data_file => [datadir("classification_samples",
    # "2dim_50:[0.0, 0.0]:[0.5 0.1; 0.1 0.2]_50:[0.0, 0.0]:[0.5 0.1; 0.1 0.2].bson")
    #                      ],
)

run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
