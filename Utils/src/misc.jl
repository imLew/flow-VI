using Distances
using Plots
using Distributions
using Random
using KernelFunctions
using LinearAlgebra
using Zygote
using ForwardDiff
using PDMats

export KL_integral
export get_pdmat
export geometric_step_size_cb
export filter_by_dict

## step_size utils
function geometric_step_size_cb(step_size, iter, factor, cutoff)
    if iter < cutoff
       return step_size * factor^iter 
   end
   return step_size * factor^cutoff
end

## math utils
function KL_integral(hist, key=:RKHS_norm)
    cumsum(get(hist, :step_sizes)[2] .* get(hist, key)[2])
end

function get_pdmat(K)
    Kmax =maximum(K)
    α = eps(eltype(K))
    while !isposdef(K+α*I) && α < 0.01*Kmax
        α *= 2.0
    end
    if α >= 0.01*Kmax
        throw(ErrorException("""Adding noise on the diagonal was not 
                             sufficient to build a positive-definite 
                             matrix:\n\t- Check that your kernel parameters 
                             are not extreme\n\t- Check that your data is 
                             sufficiently sparse\n\t- Maybe use a different 
                             kernel"""))
    end
    return PDMat(K+α*I)
end 

function data_filename(d)
    "$(d[:n_particles])particles_mu0=$(d[:μ₀])_S0=$(d[:Σ₀])_mup=$(d[:μₚ])_Sp=$(d[:Σₚ])_$(d[:n_iter])iter_stepsize=$(d[:step_size])"
end

function filter_by_key(key, values, data_array)
    out = []
    for d in data_array
        if d[key] ∈ values
            push!(out, d)
        end
    end
    return out
end

function filter_by_dict(dict, data_array)
    out = data_array
    for (k, v) in dict
        out = filter_by_key(k, v, out)
    end
    return out 
end

# flatten_index(i, j, j_max) = j + j_max *(i-1)

# function flatten_tensor(K)
#     d_max, l_max, i_max, j_max = size(K)
#     K_flat = Matrix{Float64}(undef, d_max*i_max, l_max*j_max)
#     for d in 1:d_max
#         for l in 1:l_max
#             for i in 1:i_max
#                 for j in 1:j_max
#                     K_flat[ flatten_index(d, i, i_max), 
#                             flatten_index(l, j, j_max) ] = K[d,l,i,j]
#                 end
#             end
#         end
#     end
#     return K_flat
# end

# struct MatrixKernel <: KernelFunctions.Kernel end

# function flat_matrix_kernel_matrix(k::Kernel, q)
#     d, n = size(q)
#     # kmat = Array{Float64}{undef, d, d, n, n}
#     kmat = zeros(d,d,n,n)
#     for (i, x) in enumerate(eachcol(q))
#         for (j, y) in enumerate(eachcol(q))
#             kmat[:,:,i,j] = k(x,y) .* I(d)
#         end
#     end
#     get_pdmat(flatten_tensor(kmat))
# end

# function grad_logp(d::Distribution, x)
#     if length(x) == 1
#         g = Zygote.gradient(x->log(pdf.(d, x)[1]), x )[1]
#         if isnothing(g)
#             @info "x" x
#             println("gradient nothing")
#             g = 0
#         end
#         return g
#     end
#     ForwardDiff.gradient(x->log(pdf(d, x)), reshape(x, length(x)) )
# end
