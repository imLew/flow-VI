export LogisticRegression
module LogisticRegression

using Random
using Distributions

module Structs
    struct Data
        t  # true value / target
        x  # input data
        z  # bias basis function
        function Data(t, x)
            new(t, x, [ones(size(x)[1]) x])
        end
    end
end  # Structs
Data = Structs.Data
Base.getindex(d::Data, i::Int) = Data(d.t[i], d.x[i,:]')
Base.display(d::Data) = display([d.t d.x])
Base.length(d::Data) = length(d.t)
# Base.iterate(d::Data) = (d[1], 1)
# function Base.iterate(d::Data, state)
#     if state < length(d)
#         (d[state+1], state+1)
#     else
#         return nothing
#     end
# end

σ(a) = 1 / (1 + exp(-a))

y(w) = z -> σ.(z*w)
y(D::Data, w) = y(w)(D.z)

function likelihood(D::Data, w)
    prod( y(D, w).^D.t .* (1 .-y(D, w)).^(1 .-D.t) )
end

function log_likelihood(D::Data, w)
    sum( D.t .* log.(y(D, w)) .+ (1 .- D.t) .* log.(1 .- y(D,w)) )
end

grad_log_likelihood(D::Data, w) = sum((D.t .- y(D,w)).*D.z, dims=1)'

function generate_2class_samples_from_gaussian(;n₀=5, n₁=5, μ₀=[0], μ₁=[1], Σ₀=[1], Σ₁=[1], rng=Random.GLOBAL_RNG)
    generate_2class_samples(n₀, n₁, MvNormal(μ₁, Σ₁), MvNormal(μ₀, Σ₀))
end

function generate_2class_samples(n₀, n₁, dist₀, dist₁, rng=Random.GLOBAL_RNG)
    return Data([ones(Int(n₁)); zeros(Int(n₀))], [rand(rng, dist₁, n₁)'; rand(rng, dist₀, n₀)'])
end

end # logistic regressionmodule
