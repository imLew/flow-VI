using DrWatson
using ProgressMeter
using Statistics
using ValueHistories
using KernelFunctions
using LinearAlgebra
using Random
using Flux
using Zygote
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
`:naive_WAG`. 
"""
function svgd_fit(q, grad_logp; kernel, n_iter=100, step_size=1, n_particles=50, 
                  callback=nothing, kwargs...)
    kwargs = Dict(kwargs...)
    dKL_estimator = get!(kwargs, :dKL_estimator, :RKHS_norm)
    kernel_cb! = get!(kwargs, :kernel_cb, nothing)
    step_size_cb = get!(kwargs, :step_size_cb, nothing)
    update_method = get!(kwargs, :update_method, :forward_euler)
    α = get!(kwargs, :α, false)
    c₁ = get!(kwargs, :c₁, false)
    c₂ = get!(kwargs, :c₂, false)
    if update_method == :naive_WNES && ( float(c₁) <= 0 || float(c₂) <= 0 )
        throw(ArgumentError(α, "WNES updates require c₁, c₂ ∈ ℝ."))
    end
    if update_method == :naive_WAG && float(α) <= 3
        throw(ArgumentError(α, "WAG updates require α>3."))
    end

    hist = MVHistory()
    y = copy(q) 
    ϕ = zeros(size(q))
    for i in 1:n_iter
        isnothing(kernel_cb!) ? nothing : kernel_cb!(kernel, q)
        ϵ = isnothing(step_size_cb) ? step_size : step_size_cb(step_size, i)
        update!(Val(update_method), q, ϕ, ϵ, i, kernel, grad_logp, y=y; kwargs...)
        push_to_hist!(hist, q, ϵ, ϕ, i, kernel; kwargs...)
        if !isnothing(callback)
            callback(;hist=hist, q=q, ϵ=ϵ, ϕ=ϕ, i=i, y=y, kernel=kernel, grad_logp=grad_logp, kwargs...)
        end
    end
    return q, hist
end

function push_to_hist!(hist, q, ϵ, ϕ, i, kernel; kwargs...)
    @unpack dKL_estimator = kwargs
    push!(hist, :step_sizes, i, ϵ)
    push!(hist, :ϕ_norm, i, mean(norm(ϕ)))  # save average vector norm of phi
    if typeof(dKL_estimator) == Symbol
        push!(hist, dKL_estimator, i, compute_dKL(Val(dKL_estimator), kernel, q, ϕ=ϕ))
    elseif typeof(dKL_estimator) == Array{Symbol,1}
        for estimator in dKL_estimator
            push!(hist, estimator, i, compute_dKL(Val(estimator), kernel, q, ϕ=ϕ))
        end
    end
    push!(hist, :kernel_width, kernel.transform.s)
end

function update!(::Val{:naive_WNES}, q, ϕ, ϵ, iter, kernel, grad_logp; kwargs...)
    @unpack c₁, c₂, y = kwargs
    ϕ .= calculate_phi_vectorized(kernel, y, grad_logp)
    q_new = y .+ ϵ*ϕ
    y .= q_new .+ c₁*(c₂ - 1) * (q_new .- q)
    q .= q_new
end

function update!(::Val{:naive_WAG}, q, ϕ, ϵ, iter, kernel, grad_logp; kwargs...)
    @unpack α, y = kwargs
    ϕ .= calculate_phi_vectorized(kernel, y, grad_logp)
    q_new = y .+ ϵ*ϕ
    y .= q_new .+ (iter-1)/iter .* (y.-q) + (iter + α -2)/iter * ϵ * ϕ
    q .= q_new
end

function update!(::Val{:forward_euler}, q, ϕ, ϵ, iter, kernel, grad_logp; kwargs...)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp)
    q .+= ϵ*ϕ
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

function calculate_phi_vectorized(kernel, q, grad_logp)
    n = size(q)[end]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    grad_k = kernel_grad_matrix(kernel, q)
    glp_mat = hcat( grad_logp.(eachcol(q))... )
    if n == 1  
        ϕ = glp_mat * k_mat 
    else
        ϕ =  1/n * ( glp_mat * k_mat + hcat( sum(grad_k, dims=2)... ) )
    end
end

function compute_dKL(::Val{:KSD}, kernel::Kernel, q; grad_logp)
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
    dKL /= n^2
end

function compute_dKL(::Val{:UKSD}, kernel::Kernel, q; grad_logp)
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
    dKL /= n*(n-1)
end

function compute_dKL(::Val{:RKHS_norm}, kernel::Kernel, q; ϕ)
    if size(q)[1] == 1
        invquad(kernelpdmat(kernel, q), vec(ϕ))
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
            return norm
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
	mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
end

function kernel_grad_matrix(kernel::TransformedKernel{SqExponentialKernel}, q)
    if size(q)[end] == 1
        return 0
    end
    function kernel_gradient(k::TransformedKernel{SqExponentialKernel}, x, y)
        h = 1/k.transform.s[1]^2
        -2/h * (x-y) * exp(-h\norm(x-y))
    end
    hcat(map(y->map(x->kernel_gradient(kernel, x, y), eachcol(q)), eachcol(q))...)
end
