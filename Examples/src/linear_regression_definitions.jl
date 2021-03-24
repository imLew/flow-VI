export LinearRegression
module LinearRegression

using Plots
using Distributions
using LinearAlgebra
using Optim

module Data
    struct RegressionModel
        ϕ  # feature maps
        w  # coefficients of feature maps
        β  # noise precision
    end
    struct RegressionData
        x  # input
        t  # target
    end
end
RegressionModel = Data.RegressionModel
RegressionData = Data.RegressionData

Base.getindex(d::RegressionData, i::Int) = RegressionData(d.x[i], d.t[i])
Base.display(d::RegressionData) = display([d.x d.t])
Base.length(d::RegressionData) = length(d.t)
Base.iterate(d::RegressionData) = (d[1], 1)
function Base.iterate(d::RegressionData, state)
    if state < length(d)
        (d[state+1], state+1)
    else
        return nothing
    end
end
y(model::RegressionModel) = x -> dot(model.w, model.ϕ(x))
y(model::RegressionModel, x) = y(model)(x)

# util functions for analytical solution
# returns an array (indexed by x) of arrays containing ϕ(x)
Φ(ϕ, X) = vcat( ϕ.(X)'... )
Φ(m::RegressionModel, X) = Φ(m.ϕ, X) 
# accuracy = inverse of variance
function posterior_accuracy(ϕ, β, X, Σ₀)
    inv(Σ₀) + β * Φ(ϕ, X)'Φ(ϕ, X)
end
function posterior_variance(ϕ, β, X, Σ₀)
    inv(posterior_accuracy(ϕ, β, X, Σ₀))
end
function posterior_mean(ϕ, β, D, μ₀, Σ₀)
    posterior_variance(ϕ, β, D.x, Σ₀) * ( inv(Σ₀)μ₀ + β * Φ(ϕ, D.x)' * D.t )
end
function regression_logZ(Σ₀, β, ϕ, X)
    2 \ log( det( 2π * posterior_variance(ϕ, β, X, Σ₀) ) ) 
end

function true_gauss_expectation(d::MvNormal, m::RegressionModel, D::RegressionData)
    X = reduce(hcat, m.ϕ.(D.x))
    0.5 * m.β * (tr((mean(d) * mean(d)' + cov(d)) * X * X')
        - 2 * D.t' * X' * mean(d)
        + D.t' * D.t
        + length(D) * log(m.β / 2π))
end

function generate_samples(;model::RegressionModel, n_samples=100, 
                          sample_range=[-10, 10])
    samples =  rand(Uniform(sample_range...), n_samples) 
    noise = rand(Normal(0, 1/sqrt(model.β)), n_samples)
    target = y(model).(samples) .+ noise
    return RegressionData(samples, target)
end

function likelihood(D::RegressionData, model::RegressionModel)
    prod( D-> pdf( Normal(y(model, D.x), 1/sqrt(model.β)), D.t), D )
end

E(D, model) = 2 \ sum( (D.t .- y(model).(D.x)).^2 )
function log_likelihood(D::RegressionData, model::RegressionModel)
    length(D)/2 * log(model.β/2π) - model.β * E(D, model)
end

function grad_log_likelihood(D::RegressionData, model::RegressionModel) 
    model.β * sum( ( D.t .- y(model).(D.x) ) .* model.ϕ.(D.x) )
end

end  # module
