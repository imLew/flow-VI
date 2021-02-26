export LogisticRegression
module LogisticRegression

using Distributions

σ(a) = 1 / (1 + exp(-a))

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
y(D::Data, w) = σ.(D.z*w)

function log_likelihood(D, w)
    t = D[:,1]
    x = D[:,2:end]
    z = [ones(size(x)[1]) x]
    sum( σ.(z*w) )
end

grad_logp(D, w) = sum((D.t .- y(D,w)).*D.z, dims=1)'

function generate_2class_samples_from_gaussian(;n₀, n₁, μ₀=0, μ₁=1, Σ₀=1, Σ₁=1, n_dim=1)
    generate_2class_samples(n₀, n₁, MvNormal(μ₁, Σ₁), MvNormal(μ₀, Σ₀))
end

function generate_2class_samples(n₀, n₁, dist_0, dist_1)
    return Data([ones(Int(n₁)); zeros(Int(n₀))], [rand(dist_1, n₁)'; rand(dist_0, n₀)'])
end

end # logistic regression module
