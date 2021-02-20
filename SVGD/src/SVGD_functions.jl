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

grad(f,x,y) = gradient(f,x,y)[1]
Base.identity(args...) = Base.identity(args[1])

function svgd_fit(q, grad_logp; kernel, n_iter=100, step_size=1,
        norm_method=:RKHS_norm, n_particles=50,
        kernel_cb=identity, step_size_cb=identity, update_method=:forward_euler, 
        kwargs...)
    RKHS_norm = get!(kwargs, :RKHS_norm, false)
    # SD_norm = get!(kwargs, :SD_norm, false)
    # USD_norm = get!(kwargs, :USD_norm, false)
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
    update_method in [:naive_WAG, :naive_WNES] ? y = similar(q) : nothing
    @showprogress for i in 1:n_iter
        kernel = kernel_cb(kernel, q)
        ϵ = step_size_cb(step_size, i)
        update!(Val(update_method), q, ϵ, i, kernel, grad_logp, hist, y=y, kwargs...)

        push!(hist, :step_sizes, i, ϵ)
        # if USD_norm  # save unbiased stein discrep
        #     push!(hist, :dKL_unbiased, i, 
        #           compute_dKL(Val(norm_method), kernel, q, grad_logp=grad_logp)
        #          )
        # end
        # if SD_norm  # save stein discrep
        #     push!(hist, :dKL_stein_discrep, i, 
        #           compute_dKL(Val(norm_method), kernel, q, grad_logp=grad_logp)
        #          )
        # end
        if RKHS_norm  # save rkhs norm
            push!(hist, :dKL_rkhs, i, 
                  compute_dKL(Val(norm_method), kernel, q, ϕ=ϕ)
                 )
        end
    end
    return q, hist
end

function update!(::Val{:naive_WNES}, q, ϵ, iter, kernel, grad_logp, hist; iter, kwargs...)
    @unpack c₁, c₂, y = kwargs
    ϕ = calculate_phi_vectorized(kernel, y, grad_logp)
    push!(hist, :ϕ_norm, iter, mean(norm(ϕ)))  # save average vector norm of phi
    q_new = y .+ ϵ*ϕ
    y .= q_new .+ c₁(c₂ - 1) * (q_new .- q)
    q = q_new
end

function update!(::Val{:naive_WAG}, q, ϵ, iter, kernel, grad_logp, hist; iter, kwargs...)
    @unpack α, y = kwargs
    ϕ = calculate_phi_vectorized(kernel, y, grad_logp)
    push!(hist, :ϕ_norm, iter, mean(norm(ϕ)))  # save average vector norm of phi
    q_new = y .+ ϵ*ϕ
    y .= q_new .+ (iter-1)/iter .* (y.-q) + (iter + α -2)/iter * ϵ * ϕ
    q = q_new
end

function update!(::Val{:forward_euler}, q, ϵ, iter, kernel, grad_logp, hist; kwargs...)
    ϕ = calculate_phi_vectorized(kernel, q, grad_logp)
    push!(hist, :ϕ_norm, iter, mean(norm(ϕ)))  # save average vector norm of phi
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
	mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
end

