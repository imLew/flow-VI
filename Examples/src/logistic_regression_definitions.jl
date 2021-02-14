using Distributions

module LogisticRegression

    function generate_2class_samples_from_gaussian(;n₀, n₁, μ₀=0, μ₁=1, Σ₀=1, Σ₁=1, n_dim=1)
        if n_dim == 1
            generate_2class_samples(n₀, n₁, Normal(μ₁, Σ₁), Normal(μ₀, Σ₀))
        else
            generate_2class_samples(n₀, n₁, MvNormal(μ₁, Σ₁), MvNormal(μ₀, Σ₀))
        end
    end

    function generate_2class_samples(n₀, n₁, dist_0, dist_1)
        return [ones(Int(n₁)) rand(dist_1, n₁)'; zeros((n₀)) rand(dist_0, n₀)']
    end

    σ(a) = 1 / (1 + exp(-a))

    function logistic_log_likelihood(D, w)
        t = D[:,1]
        x = D[:,2:end]
        z = [ones(size(x)[1]) x]
        sum( σ.(z*w) )
    end

    function logistic_grad_logp(D, w)
        y = D[:,1]
        x = D[:,2:end]
        z = [ones(size(x)[1]) x]
        sum((y .- σ.(z*w)).*z, dims=1)'
    end

end # module

export LogisticRegression
