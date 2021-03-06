using Random
using BSON
using DrWatson
using Distributions
using Optim
using LinearAlgebra
using ProgressMeter
using ThermodynamicIntegration

using SVGD
using Utils
using Examples
LinReg = Examples.LinearRegression
LogReg = Examples.LogisticRegression

export run_single_instance
export cmdline_run
export run_svgd
export therm_integration
export getMAP!
export therm_integration

function getMAP!(problem_params, logp, grad_logp!, D=nothing)
    problem_params[:μ_initial] = (
        if problem_params[:problem_type] == :linear_regression
            LinReg.posterior_mean(problem_params[:ϕ], problem_params[:true_β], D,
                           problem_params[:μ_prior], problem_params[:Σ_prior])
        elseif problem_params[:problem_type] == :logistic_regression
            Optim.maximizer(
                Optim.maximize(logp, grad_logp!, problem_params[:μ_prior],
                               LBFGS())
               )
        elseif problem_params[:problem_type] == :gauss_mixture_sampling
            Optim.maximizer(
                Optim.maximize(logp, grad_logp!, problem_params[:μ_initial],
                               LBFGS())
               )
        end
   )
end

function getLaplace!(p, logp, grad_logp!, D=nothing)
    getMAP!(p, logp, grad_logp!, D)
    if p[:problem_type] == :logistic_regression
        y = LogReg.y(D, p[:μ_initial])
        p[:Σ_initial] = inv(Symmetric( inv( p[:Σ_prior] )
                                      .+ D.z' * (y.*(1 .- y) .* D.z)
                                     ))
    elseif p[:problem_type] == :linear_regression
        p[:Σ_initial] = LinReg.posterior_variance(p[:ϕ], p[:true_β], D.x,
                                                  p[:Σ_prior])
    end
end

function run_svgd(::Val{:gauss_to_gauss} ;problem_params, alg_params,
                  DIRNAME="", save=true)
    svgd_results = []
    svgd_hist = MVHistory[]

    initial_dist = MvNormal(problem_params[:μ₀], problem_params[:Σ₀])
    target_dist = MvNormal(problem_params[:μₚ], problem_params[:Σₚ])

    failed_count = 0
    for i in 1:alg_params[:n_runs]
        @info "Run $i/$(alg_params[:n_runs])"
        try
            q, hist = SVGD.svgd_sample_from_known_distribution(
                        initial_dist, problem_params=problem_params,
                        target_dist; alg_params=alg_params )

            push!(svgd_results, q)
            push!(svgd_hist, hist)
        catch e
            failed_count += 1
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
    end

    true_logZ = logZ(target_dist)
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V(initial_dist, target_dist)
    estimated_logZ = if typeof(alg_params[:dKL_estimator]) <: Symbol
        [est[end] for est in estimate_logZ(H₀, EV, svgd_hist; alg_params...)]
    elseif alg_params[:update_method] ∉ [:WNES, :WAG]
        nothing
    elseif alg_params[:update_method] == :scalar_Adam
        [est[end] for est in estimate_logZ(H₀, EV, svgd_hist; alg_params...)]
    elseif typeof(alg_params[:dKL_estimator]) <: Array{Symbol,1}
            d = Dict()
            for estimator in alg_params[:dKL_estimator]
                d[estimator] = [
                    est[end] for est
                    in estimate_logZ(H₀, EV, svgd_hist;
                                     alg_params...)[1][estimator]
                   ]
            end
            d
    end
    results = merge(alg_params, problem_params,
                    @dict(true_logZ, svgd_results, svgd_hist, failed_count,
                          estimated_logZ)
                   )
    if save
        file_prefix = savename( merge(problem_params, alg_params) )
        tagsave(gdatadir(DIRNAME, file_prefix * ".bson"), results, safe=true)
    end
    return results
end

function run_svgd(::Val{:gauss_mixture_sampling} ;problem_params, alg_params,
                  DIRNAME="", save=true)
    svgd_results = []
    svgd_hist = MVHistory[]

    target_dist = MixtureModel(MvNormal, [zip(problem_params[:μₚ], problem_params[:Σₚ])...])

    grad_logp(x) = ForwardDiff.gradient(x′->log(pdf(target_dist, x′)), x)
    grad_logp!(g, x) = g .= grad_logp(x)
    if get!(problem_params, :MAP_start, false)
        getMAP!(problem_params, x->log(pdf(target_dist, x)), grad_logp!)
    end
    # if get!(problem_params, :Laplace_start, false)
    #     getLaplace!(problem_params, x->logpdf(target_dist, x), grad_logp!)
    #     problem_params[:MAP_start] = true
    # end

    initial_dist = MvNormal(problem_params[:μ_initial], problem_params[:Σ_initial])
    failed_count = 0
    for i in 1:alg_params[:n_runs]
        try
            @info "Run $i/$(alg_params[:n_runs])"
            q = rand( initial_dist, alg_params[:n_particles] )

            q, hist = svgd_fit(q, grad_logp, problem_params=problem_params; alg_params...)

            push!(svgd_results, q)
            push!(svgd_hist, hist)
        catch e
            failed_count += 1
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
    end

    true_logZ = logZ(target_dist)
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V(initial_dist, target_dist)
    estimated_logZ =
    if alg_params[:update_method] ∈ [:WNES, :WAG] || :dKL_estimator ∉ keys(alg_params)
        nothing
    elseif typeof(alg_params[:dKL_estimator]) <: Symbol
        [est[end] for est in estimate_logZ(H₀, EV, svgd_hist; alg_params...)]
    elseif typeof(alg_params[:dKL_estimator]) <: Array{Symbol,1}
        d = Dict()
        for estimator in alg_params[:dKL_estimator]
            d[estimator] = [
                est[end] for est
                in estimate_logZ(H₀, EV, svgd_hist; alg_params...)[1][estimator]
               ]
        end
        d
    end

    results = merge(alg_params, problem_params,
                    @dict(true_logZ, svgd_results, svgd_hist, failed_count, estimated_logZ)
                   )
    if save
        file_prefix = get_savename(merge(problem_params, alg_params))
        tagsave(gdatadir(DIRNAME, file_prefix * ".bson"), results,
                safe=true, storepatch = false)
    end
    return results
end

function run_svgd(::Val{:linear_regression}; problem_params, alg_params,
                  DIRNAME="", save=true)

    true_model = LinReg.RegressionModel(problem_params[:true_ϕ],
                                 problem_params[:true_w],
                                 problem_params[:true_β])
    D = LinReg.generate_samples(model=true_model,
                         n_samples=problem_params[:n_samples],
                         sample_range=problem_params[:sample_range]
                        )

    if haskey(problem_params, :therm_params)
        therm_logZ = therm_integration(problem_params, D; problem_params[:therm_params]...)
        if haskey(problem_params, :random_seed)
            Random.seed!(Random.GLOBAL_RNG, problem_params[:random_seed])
            @info "GLOBAL_RNG random seed set again because thermodynamic integration used up randomness" problem_params[:random_seed]
        end
    else
        therm_logZ = nothing
    end

    function logp(w)
        model = LinReg.RegressionModel(problem_params[:ϕ], w,
                                       problem_params[:true_β])
        ( LinReg.log_likelihood(D, model)
         + logpdf(MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior]), w)
        )
    end
    function grad_logp(w)
        model = LinReg.RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        ( LinReg.grad_log_likelihood(D, model)
         .- inv(problem_params[:Σ_prior]) * (w-problem_params[:μ_prior])
        )
    end
    grad_logp!(g, w) = g .= grad_logp(w)

    if get(problem_params, :MAP_start, false)
        getMAP!(problem_params, logp, grad_logp!, D)
    end
    if get(problem_params, :Laplace_start, false)
        @info "using Laplace_start"
        getLaplace!(problem_params, logp, grad_logp!, D)
        problem_params[:Σ_initial] *= problem_params[:Laplace_factor]
    end

    svgd_results = []
    svgd_hist = MVHistory[]

    initial_dist = MvNormal(problem_params[:μ_initial],
                            problem_params[:Σ_initial])
    failed_count = 0
    for i in 1:alg_params[:n_runs]
        try
            @info "Run $i/$(alg_params[:n_runs])"
            q = rand(initial_dist, alg_params[:n_particles])
            q, hist = svgd_fit(q, grad_logp, problem_params=problem_params,
                               D=D; alg_params...)

            push!(svgd_results, q)
            push!(svgd_hist, hist)
        catch e
        failed_count += 1
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
    end

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V(initial_dist, w -> -logp(w))
    estimated_logZ = [est[end] for est in estimate_logZ(H₀, EV, svgd_hist)]

    true_logZ = LinReg.regression_logZ(problem_params[:Σ_prior], true_model.β,
                                       true_model.ϕ, D.x)

    results = merge(alg_params, problem_params,
                    @dict(true_logZ, estimated_logZ, therm_logZ,
                          svgd_results, svgd_hist, D, failed_count))

    if save
        file_prefix = get_savename(merge(problem_params, alg_params))
        tagsave(gdatadir(DIRNAME, file_prefix * ".bson"), results,
                safe=true, storepatch = false)
    end
    return results
end

function run_svgd(::Val{:logistic_regression} ;problem_params, alg_params,
                  DIRNAME="", save=true)
    if haskey(problem_params, :sample_data_file)
        @info "Using data from file, make sure the problem params are correct"
        D = BSON.load(problem_params[:sample_data_file])[:D]
    else
        Random.seed!(Random.GLOBAL_RNG, problem_params[:random_seed])
        @info "Reset GLOBAL_RNG for sample data generation."
        D = LogReg.generate_2class_samples_from_gaussian(
                        n₀=problem_params[:n₀], n₁=problem_params[:n₁],
                        μ₀=problem_params[:μ₀], μ₁=problem_params[:μ₁],
                        Σ₀=problem_params[:Σ₀], Σ₁=problem_params[:Σ₁],
                       )
    end

    if haskey(problem_params, :therm_params)
        therm_logZ = therm_integration(problem_params, D; problem_params[:therm_params]...)
        if haskey(problem_params, :random_seed)
            Random.seed!(Random.GLOBAL_RNG, problem_params[:random_seed])
            @info "GLOBAL_RNG random seed set again because thermodynamic integration used up randomness" problem_params[:random_seed]
        end
    else
        therm_logZ = nothing
    end

    # arrays to hold results
    svgd_hist = MVHistory[]
    svgd_results = []

    function logp(w)
        ( LogReg.log_likelihood(D, w)
          + logpdf(MvNormal(problem_params[:μ_prior],
                            problem_params[:Σ_prior]),
                   w)
        )
    end
    function grad_logp(w)
        vec( LogReg.grad_log_likelihood(D, w)
             .- inv(problem_params[:Σ_prior]) * (w-problem_params[:μ_prior])
            )
    end
    grad_logp!(g, w) = g .= grad_logp(w)

    if get(problem_params, :MAP_start, false)
        getMAP!(problem_params, logp, grad_logp!, D)
    end
    if get(problem_params, :Laplace_start, false)
        getLaplace!(problem_params, logp, grad_logp!, D)
        problem_params[:MAP_start] = true
    end

    initial_dist = MvNormal(problem_params[:μ_initial],
                            problem_params[:Σ_initial])
    failed_count = 0
    Threads.@threads for i in 1:alg_params[:n_runs]
        try
            @info "Run $i/$(alg_params[:n_runs])"
            q = rand(initial_dist, alg_params[:n_particles])
            q, hist = svgd_fit(q, grad_logp, problem_params=problem_params;
                               D=D, alg_params...)

            push!(svgd_results, q)
            push!(svgd_hist, hist)
        catch e
        failed_count += 1
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
    end

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V(initial_dist, w -> -logp(w))
    if alg_params[:update_method] ∉ [:WAG, :WNES]
        estimated_logZ = [est[end] for est in estimate_logZ(H₀, EV, svgd_hist)]
    else
        estimated_logZ = nothing
    end

    results = merge(alg_params, problem_params,
                    @dict(estimated_logZ, svgd_results, svgd_hist,
                          D, therm_logZ,  failed_count))

    if save
        file_prefix = get_savename(merge(problem_params, alg_params))
        tagsave(gdatadir(DIRNAME, file_prefix * ".bson"), results,
                safe=true, storepatch = false)
    end
    return results
end

function run_svgd(;problem_params, alg_params, DIRNAME="", save=true)
    problem_params = copy(problem_params)
    alg_params = copy(alg_params)
    if DIRNAME=="" && save
        throw(ArgumentError("Cannot save to empty DIRNAME"))
    end
    if haskey(problem_params, :random_seed)
        Random.seed!(Random.GLOBAL_RNG, problem_params[:random_seed])
        @info "GLOBAL_RNG random seed set" problem_params[:random_seed]
    end
    run_svgd(Val(problem_params[:problem_type]), problem_params=problem_params,
        alg_params=alg_params, DIRNAME=DIRNAME, save=save)
end

function therm_integration(problem_params::Dict, D; nSamples=3000, nSteps=30)
    prior = MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior])
    logprior(θ) = logpdf(prior, θ)
    loglikelihood(θ) = (
        if problem_params[:problem_type] == :logistic_regression
            LogReg.log_likelihood(D, θ)
        elseif problem_params[:problem_type] == :linear_regression
            LinReg.log_likelihood(D,
                LinReg.RegressionModel(problem_params[:ϕ], θ,
                                       problem_params[:true_β])
               )
        end
       )
    alg = ThermInt(n_steps=nSteps, n_samples=nSamples)
    logZ = alg(logprior, loglikelihood, rand(prior))
end

function therm_integration(data; nSamples=3000, nSteps=30)
    prior = MvNormal(data[:μ_prior], data[:Σ_prior])
    logprior(θ) = logpdf(prior, θ)
    loglikelihood(θ) = (
        if data[:problem_type] == :logistic_regression
            LogReg.log_likelihood(data[:D], θ)
        elseif data[:problem_type] == :linear_regression
            LinReg.log_likelihood(data[:D],
                LinReg.RegressionModel(data[:ϕ], θ,
                                       data[:true_β])
               )
        end
       )
    alg = ThermInt(n_steps=nSteps, n_samples=nSamples)
    logZ = alg(logprior, loglikelihood, rand(prior))
end

function cmdline_run(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
    if length(ARGS) == 0
        run_on_gridengine(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
    elseif ARGS[1] == "make-dicts"
    # make dictionaries in a tmp directory containing the parameters for
    # all the experiment we want to run
    # also saves a dictionary mapping numbers 1 through #dicts to the dictionary
    # names to index them
        make_dicts(PROBLEM_PARAMS, ALG_PARAMS)
    elseif ARGS[1] == "run"
    # run the algorithm on the params specified in the second argument (bson)
        run_file(DIRNAME)
    elseif ARGS[1] == "run-all"
        run_all()
    elseif ARGS[1] == "make-and-run-all"
    # make the files containig the parameter dicts and start running them immediatly
        run(`julia $PROGRAM_FILE make-dicts`)
        run(`julia $PROGRAM_FILE run-all`)
    elseif ARGS[1] == "run-single-file"
    # run all experiments defined in the script from a single cmdline call
        run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
    end
end

function run_on_gridengine(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME)
    @info "Number of tasks defined by current params" ( dict_list_count(PROBLEM_PARAMS) * dict_list_count(ALG_PARAMS) )
    if haskey(ENV, "SGE_TASK_ID")
        # for running in a job array on the gridengine cluster;
        # assumes dictionaries with parameters have been created in _research/tmp
        # and that they are indexed in tmp_dict_names.bson
        dict_o_dicts = BSON.load(
                                 projectdir("_research","tmp",
                                            BSON.load( projectdir("tmp_dict_names.bson")
                                                     )[ENV["SGE_TASK_ID"]][1]
                                           )
                                )
        @info "Sampling problem: $(dict_o_dicts[:problem_params])"
        @info "Alg parameters: $(dict_o_dicts[:alg_params])"
        @time run_svgd(problem_params=dict_o_dicts[:problem_params],
                       alg_params=dict_o_dicts[:alg_params],
                       DIRNAME)
    end
end

function make_dicts(PROBLEM_PARAMS, ALG_PARAMS)
    dnames = Dict()
    for (i, alg_params) ∈ enumerate(dict_list(ALG_PARAMS))
        for (j, problem_params) ∈ enumerate(dict_list(PROBLEM_PARAMS))
            dname = tmpsave([@dict alg_params problem_params])
            dnames["$((i-1)*dict_list_count(PROBLEM_PARAMS) + j )"] = dname
        end
    end
    bson(projectdir("_research", "tmp_dict_names.bson"), dnames)
end

function run_file(DIRNAME)
    dict_o_dicts = BSON.load(ARGS[2])
    @info "Sampling problem: $(dict_o_dicts[:problem_params])"
    @info "Alg parameters: $(dict_o_dicts[:alg_params])"
    @time run_svgd(problem_params=dict_o_dicts[:problem_params],
                   alg_params=dict_o_dicts[:alg_params],
                   DIRNAME=DIRNAME)
end

function run_all()
    files = readdir(projectdir("_research", "tmp"), join=true)
    @info "Number of tmp files to run" length(files)
    for (i, file) in collect(enumerate(files))
        @info "experiment $i out of $(length(files))"
        try
            run_file(`julia $PROGRAM_FILE run $file`)
        catch e
            println(e)
        end
    end
end

function run_single_instance(PROBLEM_PARAMS, ALG_PARAMS, DIRNAME; save=true)
    params = [ (pp, ap) for pp in dict_list(PROBLEM_PARAMS),
              ap in dict_list(ALG_PARAMS)]
    p = Progress(length(params), 50)
    Threads.@threads for (i, (pp, ap)) in collect(enumerate(params))
        @info "experiment $i out of $(length(params))"
        @show pp
        @show ap
        try
            @time run_svgd(problem_params=pp, alg_params=ap,
                           DIRNAME=DIRNAME, save=save)
        catch e
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
        next!(p)
    end
end
