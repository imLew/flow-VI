using Distributions
using LinearAlgebra
using PDMats

export expectation_V
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

function estimate_logZ(H0, EV, int_KL)
    H0 - EV + int_KL
end

function numerical_expectation(d::Distribution, f; n_samples=10000, rng=Random.GLOBAL_RNG)
    sum( f, rand(rng, d, n_samples) ) / n_samples
end

# the other numerical_expectation function applies f to each element instead
# of each col :/
function num_expectation(d::Distribution, f; n_samples=10000, rng=Random.GLOBAL_RNG)
    sum( f, eachcol(rand(rng, d, n_samples)) ) / n_samples
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
