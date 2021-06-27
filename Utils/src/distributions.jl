using Distributions
using LinearAlgebra
using ValueHistories
using PDMats

using Examples
LogReg = LogisticRegression
LinReg = LinearRegression

export expectation_V
export KL_integral
export estimate_logZ
export logZ
export numerical_expectation
export num_expectation
export pdf_potential
export integrate

# This potential is already normalized
pdf_potential(d::Distribution, x) = -logpdf(d, x)

pdf_potential(d::Exponential, x) = x / Distributions.params(d)[1]

pdf_potential(d::Normal, x) = (x-mean(d))^2 / 2var(d)

pdf_potential(d::MvNormal, x) = invquad( PDMat(cov(d)), x-mean(d) )/2

function expectation_V(initial_dist::Distribution, target_dist::Distribution)
    num_expectation( initial_dist, x -> pdf_potential(target_dist, x) )
end

function expectation_V(q::Normal, p::Normal)
    0.5 * ( var(q) / var(p) + (mean(q)-mean(p))^2/var(p) )
end

function expectation_V(q::MvNormal, p::MvNormal)
    0.5 * ( tr(inv(cov(p))*cov(q)) + invquad(PDMat(cov(p)), mean(q)-mean(p)) )
end

function expectation_V(initial_dist::Distribution, V; kwargs...)
    num_expectation(initial_dist, V; kwargs...)
end

function expectation_V(::Val{:gauss_to_gauss}, data)
    expectation_V(MvNormal(data[:μ₀], data[:Σ₀]),
                  MvNormal(data[:μₚ], data[:Σₚ])
                 )
end

function expectation_V(::Val{:gauss_mixture_sampling}, data)
    expectation_V(MvNormal(data[:μ_initial], data[:Σ_initial]),
                  MixtureModel( MvNormal, [zip(data[:μₚ], data[:Σₚ])...] )
                 )
end

function expectation_V(::Val{:linear_regression}, data; kwargs...)
    expectation_V( MvNormal(data[:μ_initial], data[:Σ_initial]),
                   w -> -LinReg.log_likelihood(data[:D],
                           LinReg.RegressionModel(data[:ϕ], w, data[:true_β])
                                             )
                        - logpdf(MvNormal(data[:μ_prior],
                                          data[:Σ_prior]), w)
                   ; kwargs...)
end

function expectation_V(::Val{:logistic_regression}, data; kwargs...)
    expectation_V(
                  MvNormal(data[:μ_initial], data[:Σ_initial]),
                  w -> (
                        -LogReg.log_likelihood(data[:D], w)
                        - logpdf(MvNormal(data[:μ_prior], data[:Σ_prior]), w)
                       )
                 ; kwargs...)
end

function expectation_V(data::Dict{Symbol,Any}; kwargs...)
    expectation_V(Val(data[:problem_type]), data; kwargs...)
end

function integrate(Δx::Array, f::Array; kwargs...)
    integration_method = get(kwargs, :integration_method, :trapz)
    if integration_method == :upper_sum
        int = cumsum( Δx .* f)
    elseif integration_method == :lower_sum
        int = cumsum( Δx[1:end-1] .* f[2:end])
    elseif integration_method == :trapz
        mean_f =  vec(mean([f[1:end-1] f[2:end]], dims=2))
        int = cumsum( Δx[1:end-1] .* mean_f )
    end
    int
end

function integrate(Δx::Number, f::Array; kwargs...)
    integrate( Δx .* ones(size(f)), f; kwargs...)
end

function KL_integral(hist::MVHistory; kwargs...)
    if get(kwargs, :update_method, false) == :naive_WNES
        out = integrate(get(hist, :step_sizes)[2], get(hist, :dKL)[2]; kwargs...)
    elseif get(kwargs, :update_method, false) == :scalar_Adam
        out = integrate(get(hist, :step_sizes)[2], get(hist, :adam_dKL)[2]; kwargs...)
    else
        dKL_estimator = get(kwargs, :dKL_estimator, :RKHS_norm)
        if typeof(dKL_estimator) == Symbol
            out = integrate(
                      get(hist, :step_sizes)[2],
                      get(hist, dKL_estimator)[2];
                      kwargs...
                     )
        elseif typeof(dKL_estimator) == Array{Symbol,1}
            out = Dict()
            for estimator in dKL_estimator
                out[estimator] = integrate(
                          get(hist, :step_sizes)[2],
                          get(hist, estimator)[2];
                          kwargs...
                         )
            end
        end
    end
    return out
end

function estimate_logZ(
    H₀::Number,
    EV::Number,
    int_KL::Union{T, Array{T}},
    ;kwargs...
) where T <: Number
    H₀ .- EV .- int_KL
end

function estimate_logZ(
    H₀::Number,
    EV::Number,
    int_KLs::Dict
    ;kwargs...
    )
    out = Dict()
    for (dKL_estimator, estimate) in int_KLs
        out[dKL_estimator] = estimate_logZ(H₀, EV, estimate)
    end
    return out
end

function estimate_logZ(H₀::Number, EV::Number, hist::MVHistory; kwargs...)
    estimate_logZ(H₀, EV, KL_integral(hist; kwargs...))
end

function estimate_logZ(
    H₀::Number, EV::Number, hist_array::Array{MVHistory},
    ;kwargs...
)
    estimates = []
    for dKL_hist in hist_array
        push!(estimates, estimate_logZ(H₀, EV, dKL_hist; kwargs...))
    end
    return estimates
end

function estimate_logZ(
    initial_dist::Distribution,
    target_dist::Distribution,
    data::Dict{Symbol,Any},
    ;kwargs...
)
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist)
    estimate_logZ(H₀, EV, data[:svgd_hist]; kwargs...)
end

function estimate_logZ(
    ::T,
    data::Dict{Symbol,Any},
    ;kwargs...
) where T <: Union{Val{:logistic_regression}, Val{:linear_regression},
                   Val{:gauss_mixture_sampling}}
    initial_dist = MvNormal(data[:μ_initial], data[:Σ_initial])
    H₀ = entropy(initial_dist)
    EV = expectation_V(data)
    estimate_logZ(H₀, EV, data[:svgd_hist]; data..., kwargs...)
end

function estimate_logZ(::Val{:gauss_to_gauss}, data::Dict{Symbol,Any}; kwargs...)
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    estimate_logZ(initial_dist, target_dist, data; data..., kwargs...)
end

function estimate_logZ(data::Dict{Symbol,Any}; kwargs...)
    estimate_logZ(Val(data[:problem_type]), data; data..., kwargs...)
end

function numerical_expectation(d::Distribution, f; n_samples=10000,
                               rng=Random.GLOBAL_RNG)
    mean([ v for v in [ f(x) for x in rand(rng, d, n_samples)] if isfinite(v)] )
end

# the other numerical_expectation function applies f to each element instead
# of each col :/
function num_expectation(d::Distribution, f; n_samples=10000,
                         rng=Random.GLOBAL_RNG)
    mean([ v for v in [ f(x) for x in eachcol(rand(rng, d, n_samples))]
          if isfinite(v)] )
    mean([ f(x) for x in eachcol(rand(rng, d, n_samples))])
end

function logZ(d::Distribution)
    @warn "log(Z) for distribution $d is not know, returning 0"
    return 0
end

function logZ(d::T) where T <: Union{Normal, MvNormal}
    - logpdf( d, Distributions.params(d)[1] )
end

function logZ(d::Exponential)
    λ = 1/Distributions.params(d)[1]
    1/λ
end

function MC_logZ(
    likelihood::Function,
    prior::Distribution
    ;n_samples=1e5,
    online=false,
    batch_size=1e5
)
    if Int(batch_size) == batch_size
        batch_size = Int(batch_size)
    end
    if Int(n_samples) == n_samples
        n_samples = Int(n_samples)
    end
    # if n_samples <= batch_size || !online
        o = mean( likelihood.(rand(prior, n_samples)) )
    # elseif online
    #     n = 0
    #     o = 0
    #     while n < n_samples
    #         o += mean( likelihood.(rand(prior, batch_size)) )
    #         n += batch_size
    #     end
    # end
    return o
end

function MC_logZ(problem_params::Dict, D; kwargs...)
    MC_logZ(θ -> LogReg.likelihood(D, θ),
            MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior]);
            kwargs...)
end

export MC_logZ
