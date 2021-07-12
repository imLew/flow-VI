using DrWatson
using ProgressMeter
using Statistics
using ValueHistories
using KernelFunctions
using LinearAlgebra
using Random
using Flux
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
Possible values for update_method are `:forward_euler`, `:WNES`,
':scalar_Adam', ':scalar_RMS_prop', ':scalar_adagrad' `:WAG`.
"""
function svgd_fit(q, grad_logp; kwargs...)
    kernel = TransformedKernel(SqExponentialKernel(), ScaleTransform(1.))
    kwargs = Dict(kwargs...)
    callback = get(kwargs, :callback, nothing)
    n_iter = get(kwargs, :n_iter, 1)
    n_particles = get(kwargs, :n_particles, 1)
    step_size = get(kwargs, :step_size, 1)
    kernel_cb! = get!(kwargs, :kernel_cb, nothing)
    step_size_cb = get!(kwargs, :step_size_cb, nothing)
    update_method = get!(kwargs, :update_method, :forward_euler)
    annealing_schedule = get!(kwargs, :annealing_schedule, nothing)
    annealing_params = get!(kwargs, :annealing_params, [])
    progress = get(kwargs, :progress, true)

    aux_vars = Dict()
    if update_method in [:scalar_adagrad, :scalar_RMS_prop]
        aux_vars[:Gₜ] = [0.]
    elseif update_method == :scalar_Adam
        aux_vars[:mₜ] = zeros(size(q))
        aux_vars[:mₜ₋₁] = zeros(size(q))
        aux_vars[:vₜ] = zeros(size(q))
        aux_vars[:𝔼∇mₜ₋₁] = [0.]
        aux_vars[:𝔼∇ϕₜ₋₁] = [0.]
    elseif update_method in [:WAG, :WNES]
        aux_vars[:y] = copy(q)
        aux_vars[:qₜ₋₁] = copy(q)
        aux_vars[:qₜ₋₂] = copy(q)
    end
    hist = MVHistory()
    ϕ = zeros(size(q))
    progress ? p = Progress(n_iter, 1) : nothing
    for i in 1:n_iter
        isnothing(kernel_cb!) ? nothing : kernel_cb!(kernel, q)
        ϵ = isnothing(step_size_cb) ? [step_size] : [step_size_cb(step_size, i)]
        γₐ = if isnothing(annealing_schedule)
            [1.]
        else
            [annealing_schedule(i, n_iter; annealing_params...)]
        end
        ∇logp_mat = mapreduce(grad_logp, hcat, eachcol(q))
        push_to_hist!(hist, q, ϵ, ϕ, i, γₐ, kernel, grad_logp, aux_vars,
                      ∇logp_mat,; kwargs...)
        update!(Val(update_method), q, ϕ, ϵ, kernel, grad_logp, aux_vars,
                        ∇logp_mat, iter=i, γₐ=γₐ; kwargs...)
        if !isnothing(callback)
            callback(;hist=hist, q=q, ϕ=ϕ, i=i, kernel=kernel, ∇logp_mat,
                     grad_logp=grad_logp, aux_vars..., kwargs...)
        end
        progress ? next!(p) : nothing
    end
    return q, hist
end

function push_to_hist!(
    hist, q, ϵ, ϕ, i, γₐ, kernel, grad_logp, aux_vars, ∇logp_mat,
    ; kwargs...
)
    push!(hist, :step_sizes, i, ϵ[1])
    push!(hist, :annealing, i, γₐ[1])
    push!(hist, :ϕ_norm, i, mean(norm(ϕ)))

    dKL_estimator = get(kwargs, :dKL_estimator, false)
    if kwargs[:update_method] ∈ [:WAG, :WNES]
        nothing
    elseif kwargs[:update_method] == :scalar_Adam
        dKL = dKL_Adam(kernel, q, ϕ, grad_logp, aux_vars, ϵ, ∇logp_mat; kwargs...)
        push!(hist, :adam_dKL, i, dKL)
    elseif typeof(dKL_estimator) == Symbol
        dKL = compute_dKL(Val(dKL_estimator), kernel, q, ∇logp_mat, ϕ=ϕ,
                          grad_logp=grad_logp)
        dKL += dKL_annealing_correction(ϕ, ∇logp_mat, q, γₐ)
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

function dKL_annealing_correction(ϕ, ∇logp_mat, q, γₐ)
    -(1-γₐ[1])*ϕ⋅∇logp_mat/size(q, 2)
end

function dKL_Adam(kernel, q, ϕ, grad_logp, aux_vars, ϵ, ∇logp_mat; kwargs...)
    β₁ = get(kwargs, :β₁, false)
    β₂ = get(kwargs, :β₂, false)
    N = size(q, 2)
    aux_vars[:𝔼∇mₜ₋₁] .= β₁ .* aux_vars[:𝔼∇mₜ₋₁] .+ (1-β₁) .* aux_vars[:𝔼∇ϕₜ₋₁]
    norm_ϕ = RKHS_norm(kernel, q, ϕ=ϕ)
    dKL = β₁.*(aux_vars[:𝔼∇mₜ₋₁].+aux_vars[:mₜ₋₁]⋅∇logp_mat/N).+(1-β₁).*norm_ϕ
    return -dKL[1]
end

function update!(::Val{:scalar_Adam},
q, ϕ, ϵ, kernel, grad_logp, aux_vars, ∇logp_mat
; kwargs...
)
    t = get(kwargs, :iter, false)
    β₁ = get(kwargs, :β₁, false)
    β₂ = get(kwargs, :β₂, false)
    stepsize_method = get(kwargs, :adam_stepsize_method, :average)
    unbiased = get(kwargs, :Adam_unbiased, false)

    # We must compute 𝔼∇ϕ here because we need it for dKL/dt later.
    aux_vars[:𝔼∇ϕₜ₋₁] .= 𝔼∇ϕ(kernel, q, grad_logp, ∇logp_mat, unbiased=unbiased)

    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    aux_vars[:mₜ₋₁] .= aux_vars[:mₜ]
    aux_vars[:mₜ] .= β₁ .* aux_vars[:mₜ] + (1-β₁) .* ϕ
    aux_vars[:vₜ] .= β₂ .* aux_vars[:vₜ] + (1-β₂) .* ϕ.^2

    if stepsize_method == :average
        ϵ .*= sqrt(1-β₂^t)./(1-β₁^t) .* mean(1.0./sqrt.(aux_vars[:vₜ]).+1)
    elseif stepsize_method == :minimum
        ϵ .*= sqrt(1-β₂^t)./(1-β₁^t) .* 1.0/sqrt.(maximum(aux_vars[:vₜ]).+1)
    end

    q .+= ϵ .* aux_vars[:mₜ]
end

function 𝔼∇ϕ(kernel, q, ∇logp, ∇logp_mat; unbiased=false)
    d, N = size(q)
    h = 1/kernel.transform.s[1]^2
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    ∇k = -1.0.*kernel_grad_matrix(kernel, q)
    # Multiply by -1 because we need the gradient, ∇, with respect to the
    # second argument and for the RBF kernel that is -1 times the gradient
    # with respect to the first argument.
    if unbiased
        𝔼∇ϕ = (
               sum(k_mat .* (d/h .- 1/h^2 .* pairwise(SqEuclidean(), q)))
               - sum(diag(k_mat .* (d/h .- 1/h^2 .* pairwise(SqEuclidean(), q))))
               +sum(∇k'*∇logp_mat) - ∇k⋅∇logp_mat
              ) / (N*(N-1))
    else
        𝔼∇ϕ = N^2\(sum(k_mat .* (d/h .- 1/h^2 .* pairwise(SqEuclidean(), q)))
                   +sum(∇k'*∇logp_mat)
                  )
    end
    return 𝔼∇ϕ
end

function update!(::Val{:scalar_RMS_prop},
q, ϕ, ϵ, kernel, grad_logp, aux_vars, ∇logp_mat
; kwargs...
)
    γ = get(kwargs, :γ, false)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    aux_vars[:Gₜ] .= γ * norm(ϕ)^2 .+ (1-γ) * aux_vars[:Gₜ]
    ϵ .= ϵ/(√(aux_vars[:Gₜ][1] + 1))
    q .+= ϵ .*ϕ
end

function update!(::Val{:scalar_adagrad},
q, ϕ, ϵ, kernel, grad_logp, aux_vars, ∇logp_mat
; kwargs...
)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    aux_vars[:Gₜ] .+= norm(ϕ)^2
    N = size(ϕ, 2)
    ϵ .= N*ϵ/(aux_vars[:Gₜ][1] + 1)
    q .+= ϵ .*ϕ
end

function update!(
::Val{:WAG}, q, ϕ, ϵ, kernel, grad_logp, aux_vars,
        ∇logp_mat;
                 kwargs...)
    # aux_vars[:qₜ₋₁] = copy(q)
    iter = get(kwargs, :iter, false)
    α = get(kwargs, :α, false)
    ϕ .= calculate_phi_vectorized(kernel, aux_vars[:y], grad_logp,
                                  ∇logp_mat; kwargs...)
    q_new = aux_vars[:y] .+ ϵ.*ϕ
    aux_vars[:y] .= q_new .+ (iter-1)/iter .* (aux_vars[:y].-q) + (iter + α -2)/iter * ϵ .* ϕ
    q .= q_new
end

function update!(
::Val{:forward_euler},
q, ϕ, ϵ, kernel, grad_logp, aux_vars, ∇logp_mat,
;kwargs...
)
    ϕ .= calculate_phi_vectorized(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    q .+= ϵ.*ϕ
end

function update!(::Val{:WNES},
q, ϕ, ϵ, kernel, grad_logp, aux_vars, ∇logp_mat
; kwargs...
)
    aux_vars[:qₜ₋₂] .= aux_vars[:qₜ₋₁]
    aux_vars[:qₜ₋₁] .= q
    ϕ .= WNes_ϕ(ϵ, q, aux_vars[:qₜ₋₁], kernel, kwargs[:c₁],
               kwargs[:c₂], grad_logp, ∇logp_mat; kwargs...)
    q .+= ϵ .* ϕ
end

function WNes_ϕ(ϵ, q, qₜ₋₁, kernel, c₁, c₂, grad_logp, ∇logp_mat; kwargs...)
    CΔq = c₁*(c₂-1).*(q.-qₜ₋₁)
    ϵ.\CΔq .+ calculate_phi_vectorized(kernel, q.+CΔq, grad_logp, ∇logp_mat
                                       ; kwargs...)
end

function divergence(F, X)
    div = 0
    for i in 1:length(X)
        x_top = X[1:i-1]
        x_bot = X[i+1:end]
        f(x) = F(vcat(x_top, x, x_bot))[i]
        div += ForwardDiff.derivative(f, X[i])
    end
    return div
end

function WNes_dKL(kernel, q, ϕ, grad_logp, aux_vars, ϵ, ∇logp_mat; kwargs...)
    # c₁ = get(kwargs, :c₁, false)
    # c₂ = get(kwargs, :c₂, false)
    # C = c₁*(c₂-1)
    # N = size(q, 2)
    # h = 1/kernel.transform.s[1]^2
    # d = size(q)[1]
    # t(x, xₜ₋₁) = (1+C).*x .- xₜ₋₁
    # y = map(t, q, aux_vars[:qₜ₋₁])
    # dKL = 0

    # k_mat = KernelFunctions.kernelmatrix(kernel, q)

    # # This sums over all combinations of particles, not sure whether that is
    # # correct, but since WNes_dKL is probably not correct at all anyway it
    # # doesn't matter for now.
    # dKL += sum(ϕ'*∇logp_mat)

    # ∇k = kernel_grad_matrix(kernel, y)
    # # See comment above.
    # dKL += sum(∇k'*∇logp_mat ) / N

    # dKL += N \ sum( k_mat .* ( 2*d/h .- 4/h^2 .* pairwise(SqEuclidean(), y) ) )

    # for xₜ₋₁ in eachcol(aux_vars[:qₜ₋₁])
    #     function ϕ̂(x)
    #         CΔq = c₁*(c₂-1).*(x.-xₜ₋₁)
    #         ϵ.\CΔq .+ calculate_phi(kernel, x.+CΔq, grad_logp, ∇logp_mat;
    #                                 kwargs...)
    #     end
    #     dKL += divergence(ϕ̂, xₜ₋₁)
    # end

    # return dKL/N
    0.
end

function calculate_phi_vectorized(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    γₐ = get(kwargs, :γₐ, [1.])
    N = size(q, 2)
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    grad_k = kernel_grad_matrix(kernel, q)
    ϕ = 1/N * (γₐ .* ∇logp_mat * k_mat .+ grad_k )
end

function calculate_phi(kernel, q, grad_logp, ∇logp_mat; kwargs...)
    # glp = grad_logp.(eachcol(q))
    @warn "Using ∇logp_mat here is untested."
    glp = ∇logp_mat
    ϕ = zero(q)
    for (i, xi) in enumerate(eachcol(q))
        for (xj, glp_j) in zip(eachcol(q), glp)
            ϕ[:, i] .+= kernel(xj, xi) * glp_j .+ kernel_gradient(kernel, xj, xi)
            # d = kernel(xj, xi) * glp_j
            # K = kernel_gradient( kernel, xj, xi )
            # ϕ[:, i] .+= d .+ K
        end
    end
    ϕ ./= size(q, 2)
end

function compute_dKL(::Val{:KSD}, kernel::Kernel, q, ∇logp_mat
;kwargs...
)
    d, N = size(q)
    h = 1/kernel.transform.s[1]^2
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = ∇logp_mat[:, i]
        for (j, y) in enumerate(eachcol(q))
            glp_y = ∇logp_mat[:, j]
            dKL += (
                    (glp_x .- (x.-y)./h) ⋅ (glp_y .+ (x.-y)./h) + d/h
                   ) * k_mat[i,j]
        end
    end
    -dKL / N^2
end

function compute_dKL(::Val{:uKSD}, kernel::Kernel, q, ∇logp_mat
;kwargs...
)
    d, N = size(q)
    h = 1/kernel.transform.s[1]^2
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = ∇logp_mat[:, i]
        for (j, y) in enumerate(eachcol(q))
            if i != j
                glp_y = ∇logp_mat[:, j]
                dKL += (
                        (glp_x .- (x.-y)./h) ⋅ (glp_y .+ (x.-y)./h) + d/h
                       ) * k_mat[i,j]
            end
        end
    end
    -dKL / (N*(N-1))
end

function compute_dKL(::Val{:RKHS_norm}, kernel::Kernel, q, args...; ϕ, kwargs...)
    return -RKHS_norm(kernel, q; ϕ=ϕ, kwargs...)
end

function RKHS_norm(kernel::Kernel, q; ϕ, kwargs...)
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

# not being used, double check before using another kernel
# function kernel_grad_matrix(kernel::KernelFunctions.Kernel, q)
#     if size(q)[end] == 1
#         return 0
#     end
#     grad(f,x,y) = gradient(f,x,y)[1]
# 	grad_k = mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
#     sum(grad_k, dims=2)
# end

# gradient of k(x,y) = exp(-‖x-y‖²/2h) with respect to x
function kernel_gradient(k::TransformedKernel{SqExponentialKernel}, x, y)
    - k.transform.s[1]^2 .* (x-y) .* k(x,y)
end

function kernel_grad_matrix(kernel::TransformedKernel{SqExponentialKernel}, q)
    ∇k = zeros(size(q))
    for (j, y) in enumerate(eachcol(q))
        for (i, x) in enumerate(eachcol(q))
            ∇k[:,j] .+= kernel_gradient(kernel, x, y)
        end
    end
    ∇k
end
