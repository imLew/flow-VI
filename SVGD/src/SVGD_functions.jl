using DrWatson
using ProgressMeter
using Statistics
using ValueHistories
using KernelFunctions
using LinearAlgebra
using Random
using Flux
# using Zygote
using ForwardDiff
using Distances
using PDMats

export svgd_fit
export calculate_phi_vectorized
export compute_dKL
export kernel_grad_matrix
export calculate_phi

"""
Fit the samples in q to the distribution corresponding to grad_logp.
Possible values for dKL_estimator are `:RKHS_norm`, `:KSD`, `:UKSD`; they can be
combined by putting them in array.
Possible values for update_method are `:forward_euler`, `:naive_WNES`,
':scalar_Adam', ':scalar_RMS_prop', ':scalar_adagrad' `:naive_WAG`.
"""
function svgd_fit(q, grad_logp; kernel, callback=nothing, kwargs...)
    kwargs = Dict(kwargs...)
    n_iter = get(kwargs, :n_iter, 1)
    n_particles = get(kwargs, :n_particles, 1)
    step_size = get(kwargs, :step_size, 1)
    kernel_cb! = get!(kwargs, :kernel_cb, nothing)
    step_size_cb = get!(kwargs, :step_size_cb, nothing)
    update_method = get!(kwargs, :update_method, :forward_euler)
    annealing_schedule = get!(kwargs, :annealing_schedule, nothing)
    annealing_params = get!(kwargs, :annealing_params, [])

    aux_vars = Dict()
    if update_method in [:scalar_adagrad, :scalar_RMS_prop]
        aux_vars[:Gₜ] = [0.]
    elseif update_method == :scalar_Adam
        aux_vars[:mₜ] = zeros(size(q))
        aux_vars[:vₜ] = zeros(size(q))
    elseif update_method in [:naive_WAG, :naive_WNES]
        aux_vars[:y] = copy(q)
    end
    hist = MVHistory()
    ϕ = zeros(size(q))
    for i in 1:n_iter
        isnothing(kernel_cb!) ? nothing : kernel_cb!(kernel, q)
        ϵ = isnothing(step_size_cb) ? [step_size] : [step_size_cb(step_size, i)]
        γₐ = if isnothing(annealing_schedule)
            [1.]
        else
            [annealing_schedule(i, n_iter; annealing_params...)]
        end
        update!(Val(update_method), q, ϕ, ϵ, kernel, grad_logp, aux_vars,
                iter=i, γₐ=γₐ; kwargs...)
        push_to_hist!(hist, q, ϵ, ϕ, i, γₐ, kernel, grad_logp; kwargs...)
        if !isnothing(callback)
            callback(;hist=hist, q=q, ϕ=ϕ, i=i, kernel=kernel,
                     grad_logp=grad_logp, aux_vars=aux_vars, kwargs...)
        end
    end
    return q, hist
end

function push_to_hist!(hist, q, ϵ, ϕ, i, γₐ, kernel, grad_logp; kwargs...)
    dKL_estimator = get(kwargs, :dKL_estimator, false)
    push!(hist, :step_sizes, i, ϵ[1])
    push!(hist, :annealing, i, γₐ[1])
    push!(hist, :ϕ_norm, i, mean(norm(ϕ)))  # save average vector norm of phi
    if typeof(dKL_estimator) == Symbol
        dKL = compute_dKL(Val(dKL_estimator), kernel, q, ϕ=ϕ, grad_logp=grad_logp)
        dKL += dKL_annealing_correction(ϕ, grad_logp, q, γₐ)
        push!(hist, dKL_estimator, i, dKL)
    elseif typeof(dKL_estimator) == Array{Symbol,1}
        for estimator in dKL_estimator
            dKL = compute_dKL(Val(estimator), kernel, q, ϕ=ϕ, grad_logp=grad_logp)
            dKL += dKL_annealing_correction(ϕ, grad_logp, q, γₐ)
            push!(hist, estimator, i, dKL)
        end
    end
    push!(hist, :kernel_width, kernel.transform.s)
end

function dKL_annealing_correction(ϕ, grad_logp, q, γₐ)
    c = 0
    for (xᵢ, ϕᵢ) in zip(eachcol(ϕ), eachcol(q))
        c += dot(ϕᵢ, grad_logp(xᵢ))
    end
    -(1-γₐ[1])*c
end

function calculate_phi_vectorized(kernel, q, grad_logp; kwargs...)
    γₐ = get(kwargs, :γₐ, [1.])
    n = size(q)[end]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    grad_k = kernel_grad_matrix(kernel, q)
    glp_mat = mapreduce( grad_logp, hcat, (eachcol(q)) )
    if n == 1
        ϕ = γₐ .* glp_mat * k_mat
    else
        ϕ =  1/n * (γₐ .* glp_mat * k_mat + grad_k )
    end
end

function update!(::Val{:scalar_Adam}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    iter = get(kwargs, :iter, false)
    β₁ = get(kwargs, :β₁, false)
    β₂ = get(kwargs, :β₂, false)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp; kwargs...)
    aux_vars[:mₜ] .= (β₁ .* aux_vars[:mₜ] + (1-β₁) .* ϕ)
    aux_vars[:vₜ] .= β₂ .* aux_vars[:vₜ] + (1-β₂) .* ϕ.^2
    N = size(ϕ, 1)
    ϵ .= ϵ.*sqrt((1-β₂^iter)./(1-β₁^iter)) ./ mean(sqrt.(aux_vars[:vₜ]))
    q .+= ϵ .* aux_vars[:mₜ]./(1-β₁^iter)
end

function update!(::Val{:scalar_RMS_prop}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    γ = get(kwargs, :γ, false)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp; kwargs...)
    aux_vars[:Gₜ] .= γ * norm(ϕ)^2 .+ (1-γ) * aux_vars[:Gₜ]
    N = size(ϕ, 1)
    ϵ .= N*ϵ/(aux_vars[:Gₜ][1] + 1)
    q .+= ϵ .*ϕ
end

function update!(::Val{:scalar_adagrad}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp; kwargs...)
    aux_vars[:Gₜ] .+= norm(ϕ)^2
    N = size(ϕ, 1)
    ϵ .= N*ϵ/(aux_vars[:Gₜ][1] + 1)
    q .+= ϵ .*ϕ
end

function update!(::Val{:naive_WNES}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    c₁ = get(kwargs, :c₁, false)
    c₂ = get(kwargs, :c₂, false)
    ϕ .= calculate_phi_vectorized(kernel, aux_vars[:y], grad_logp; kwargs...)
    q_new = aux_vars[:y] .+ ϵ.*ϕ
    aux_vars[:y] .= q_new .+ c₁*(c₂ - 1) * (q_new .- q)
    q .= q_new
end

function update!(::Val{:naive_WAG}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    iter = get(kwargs, :iter, false)
    α = get(kwargs, :α, false)
    ϕ .= calculate_phi_vectorized(kernel, aux_vars[:y], grad_logp; kwargs...)
    q_new = aux_vars[:y] .+ ϵ.*ϕ
    aux_vars[:y] .= q_new .+ (iter-1)/iter .* (aux_vars[:y].-q) + (iter + α -2)/iter * ϵ .* ϕ
    q .= q_new
end

function update!(::Val{:forward_euler}, q, ϕ, ϵ, kernel, grad_logp, aux_vars;
                 kwargs...)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp; kwargs...)
    q .+= ϵ.*ϕ
end

function calculate_phi(kernel, q, grad_logp)
    glp = grad_logp.(eachcol(q))
    ϕ = zero(q)
    for (i, xi) in enumerate(eachcol(q))
        for (xj, glp_j) in zip(eachcol(q), glp)
            d = kernel(xj, xi) * glp_j
            # K = kernel_gradient( kernel, xj, xi )
            K = gradient( x->kernel(x, xi), xj )[1]
            ϕ[:, i] .+= d .+ K
        end
    end
    ϕ ./= size(q)[end]
end

function compute_dKL(::Val{:KSD}, kernel::Kernel, q; grad_logp, kwargs...)
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = grad_logp(x)
        for (j, y) in enumerate(eachcol(q))
            dKL += k_mat[i,j] * dot(glp_x, grad_logp(y))
            dKL += dot( gradient(x->kernel(x,y), x)[1], grad_logp(y) )
            dKL += dot( gradient(y->kernel(x,y), y)[1], glp_x )
            dKL += k_mat[i,j] * ( 2*d/h - 4/h^2 * SqEuclidean()(x,y))
        end
    end
    -dKL / (n^2)
end

function compute_dKL(::Val{:uKSD}, kernel::Kernel, q; grad_logp, kwargs...)
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = grad_logp(x)
        for (j, y) in enumerate(eachcol(q))
            if i != j
                dKL += k_mat[i,j] * dot(glp_x, grad_logp(y))
                dKL += dot( gradient(x->kernel(x,y), x)[1], grad_logp(y) )
                dKL += dot( gradient(y->kernel(x,y), y)[1], glp_x )
                dKL += kernel(x,y) * ( 2d/h - 4/h^2 * SqEuclidean()(x,y))
            end
        end
    end
    # dKL += sum(k_mat .* ( 2*d/h .- 4/h^2 * pairwise(SqEuclidean(), q)))
    -dKL / (n*(n-1))
end

function compute_dKL(::Val{:RKHS_norm}, kernel::Kernel, q; ϕ, kwargs...)
    if size(q)[1] == 1
        - invquad(kernelpdmat(kernel, q), vec(ϕ))
    else
        # this first method tries to flatten the tensor equation
        # invquad(flat_matrix_kernel_matrix(kernel, q), vec(ϕ))
        # the second method should be the straight forward case for a
        # kernel that is a scalar f(x) times identity matrix
        norm = 0
        try
            k_mat = kernelpdmat(kernel, q)
            for f in eachrow(ϕ)
                norm += invquad(k_mat, vec(f))
            end
            return - norm
        catch e
            if e isa PosDefException
                @show kernel
            end
            rethrow(e)
        end
    end
end

function kernel_grad_matrix(kernel::KernelFunctions.Kernel, q)
    if size(q)[end] == 1
        return 0
    end
    grad(f,x,y) = gradient(f,x,y)[1]
	grad_k = mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
    sum(grad_k, dims=2)
end

function kernel_grad_matrix(kernel::TransformedKernel{SqExponentialKernel}, q)
    if size(q)[end] == 1
        return 0
    end
    function kernel_gradient(k::TransformedKernel{SqExponentialKernel}, x, y)
        - k.transform.s[1]^2 * (x-y) * k(x,y)
    end
    ∇k = zeros(size(q))
    for (j, y) in enumerate(eachcol(q))
        for (i, x) in enumerate(eachcol(q))
            ∇k[:,j] += kernel_gradient(kernel, x, y)
        end
    end
    ∇k
end

export kernel_grad_matrix
