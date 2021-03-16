using Distributions
using LinearAlgebra
using ValueHistories
using PDMats

export expectation_V
export KL_integral
export estimate_logZ
export logZ
export numerical_expectation
export num_expectation
export pdf_potential

function expectation_V(initial_dist::Distribution, target_dist::Distribution) 
    numerical_expectation( initial_dist, x -> pdf_potential(target_dist, x) )
end

function expectation_V(initial_dist::Normal, target_dist::Normal)
    μ₀, σ₀ = Distributions.params(initial_dist)
    μₚ, σₚ = Distributions.params(target_dist)
    0.5 * ( σ₀^2 / σₚ^2 + (μ₀-μₚ)^2/σₚ^2  )
end

function expectation_V(initial_dist::MvNormal, target_dist::MvNormal)
    μ₀, Σ₀ = Distributions.params(initial_dist)
    μₚ, Σₚ = Distributions.params(target_dist)
    0.5 * ( tr(inv(Σₚ)*Σ₀) + invquad(Σₚ, μ₀-μₚ) )
end

function KL_integral(hist, method=:RKHS_norm)
    cumsum(get(hist, :step_sizes)[2] .* get(hist, method)[2])[1]
end

function estimate_logZ(H₀, EV, int_KL)
    H₀ .- EV .+ int_KL
end

function estimate_logZ(H₀, EV, hist::MVHistory, method=:RKHS_norm)
    estimate_logZ(H₀, EV, KL_integral(hist, method))
end

function estimate_logZ(data::Dict{Symbol,Any}, method=:RKHS_norm)
    estimate_logZ(Val(data[:problem_type]), data, method)
end

function estimate_logZ(::Val{:gauss_to_gauss}, data::Dict{Symbol,Any}, 
                       method=:RKHS_norm)
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    estimate_logZ(initial_dist, target_dist, data, method)
end

function estimate_logZ(initial_dist::Distribution, target_dist::Distribution,
                       data::Dict{Symbol,Any}, method=:RKHS_norm)
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist)
    estimate_logZ(H₀, EV, data[:svgd_hist], method)
end

function estimate_logZ(H₀, EV, hist_array::Array{MVHistory}, method=:RKHS_norm)
    estimates = []
    for dKL_hist in hist_array
        push!(estimates, estimate_logZ(H₀, EV, dKL_hist, method))
    end
    return estimates
end        

function numerical_expectation(d::Distribution, f; n_samples=10000, rng=Random.GLOBAL_RNG)
    mean([ v for v in [ f(x) for x in rand(rng, d, n_samples)] if isfinite(v)] ) 
end

# the other numerical_expectation function applies f to each element instead
# of each col :/
function num_expectation(d::Distribution, f; n_samples=10000, rng=Random.GLOBAL_RNG)
    mean([ v for v in [ f(x) for x in eachcol(rand(rng, d, n_samples))] if isfinite(v)] ) 
end

function logZ(d::Distribution)
    println("log(Z) for distribution $d is not know, returning 0")
    return 0
end

function logZ(d::T) where T <: Union{Normal, MvNormal}
    - logpdf( d, Distributions.params(d)[1] )
end

function logZ(d::Exponential)
    λ = 1/Distributions.params(d)[1] 
    1/λ
end

function pdf_potential(d::Distribution, x)
    -logpdf(d, x) # This potential is already normalized
end

function pdf_potential(d::Exponential, x)
    # Distribution.jl uses inverse param θ=1/λ (i.e. 1/θ e^{-x/θ})
    λ = 1/Distributions.params(d)[1] 
    λ * x
end

function pdf_potential(d::Normal, x)
    μ, σ = Distributions.params(d)
    2 \ ((x-μ)/σ)^2
end

function pdf_potential(d::MvNormal, x)
    μ, Σ = Distributions.params(d)
    2 \ invquad(Σ, x-μ)
end
