using DrWatson
@quickactivate

using Utils

PROBLEM_PARAMS = Dict(
    :problem_type => :linear_regression,
    :n_samples => 10 ,
    :sample_range =>[ [-3, 3] ],
    :true_ϕ =>[ x -> [1, x, x^2] ],
    :true_w =>[ [2, -1, 0.2] ],
    :true_β => 2,
    :ϕ =>[ x -> [1, x, x^2] ],
    :μ_initial => [ zeros(3) ] ,
    :Σ_initial =>[ [1. 0 0; 0 1 0; 0 0 1] ],
    :μ_prior => [ zeros(3) ] ,
    :Σ_prior =>[ [1. 0 0; 0 1 0; 0 0 1] ],
    :MAP_start => false ,
    :Laplace_start => false,
    # :Laplace_factor => @onlyif(:Laplace_start == true, [ 10., 0.1 ]),
    :random_seed => [ 0 ],
)

ALG_PARAMS = Dict(
    :n_iter => [ 5000 ],
    :step_size => [ 0.010, 0.015 ],
    :n_particles => 50,
    :update_method => [ :forward_euler ],
    :β₁ => 0.9,
    :β₂ => 0.999,
    :γ => 0.8 ,
    :kernel_cb => median_trick_cb!,
    :dKL_estimator => :RKHS_norm,
    :n_runs => 1,
)

DIRNAME = "linear_regression/GDvariants"
run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME, save=false)
