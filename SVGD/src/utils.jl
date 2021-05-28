using Distributions
using LinearAlgebra
using Zygote
using ForwardDiff

export svgd_sample_from_known_distribution

function ∇logp(d::Distribution, x)
    if length(x) == 1
        g = Zygote.gradient(x->log(pdf.(d, x)[1]), x )[1]
        if isnothing(g)
            @info "x" x
            println("gradient nothing")
            g = 0
        end
        return g
    end
    ForwardDiff.gradient(x->log(pdf(d, x)), reshape(x, length(x)) )
end

function ∇logp(d::MvNormal, x)
    -cov(d) * (x.-mean(d))
end

function svgd_sample_from_known_distribution(initial_dist, target_dist;
                                             alg_params, kwargs...)
    glp(x) = ∇logp(target_dist, x)
    q = rand( initial_dist, alg_params[:n_particles] )
    if length(size(q)) == 1
        q = reshape(q, (1, length(q)))
    end
    q, hist = svgd_fit( q, glp; alg_params... , kwargs...)
end
