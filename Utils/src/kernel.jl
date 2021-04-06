using Zygote
using Distances
using KernelFunctions

## kernel utils
export kernel_gradient
export median_trick_cb!
export median_cb!

function kernel_gradient(k::Kernel, x, y)
    Zygote.gradient( x->k(x,y), x)[1]
end

function median_trick_cb!(kernel::Kernel, q)
    kernel.transform.s .= sqrt(2/median_trick(q))
end

function median_cb!(kernel, q)
    kernel.transform.s .= 1/median(pairwise(Euclidean(), q, dims=2))
end

function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x, dims=2)
    median(d)^2/log(size(x)[end])
end

export median_trick

function HE_loss(q, h::Number)
    D, N = size(q)
    function d(i,j)
        q[:,i] - q[:,j]
    end
    function e(i,j)
        exp( - norm(d(i,j))^2 / (2h)  - D/2 * log(h) )
    end
    function fsum(f)
        sum(f, 1:N)
    end
    function g(k)
        (
            fsum( j->e(j,k)*norm(d(j,k))^2 )
            - h * D * fsum(j->e(k,j))
            - fsum( j -> fsum(i->e(i,j))^(-1) * e(j,k)
                         * dot( d(j,k), fsum(i->e(i,j)*d(i,j)) )
                  )
        )
    end
    fsum( k -> g(k)^2  ) / (2π)^D
end

function HE_loss(q, h::Array)
    HE_loss(q, h[1])
end

export HE_loss

# function HE_loss_derivative(q, h)
#     D, N = size(q)
#     function d(i,j)
#         q[:,i] - q[:,j]
#     end
#     function e(i,j)
#         exp( - norm(d(i,j))^2 / 2h - D/2 * log(h) )
#     end
#     function fsum(f)
#         sum(f, 1:N)
#     end
#     function g(h, k)
#         (
#             fsum( j->e(k,j)*norm(d(k,j))^2 ) - h*D * fsum(j->e(k,j))
#             - fsum( j -> 1/fsum(i->e(i,j)) * e(j,k) * dot( d(j,k), fsum(i->e(i,j)*d(i,j)) ) ) 
#         )
#     end
#     function g_prime(h,k)
#         (
#             1/(2h^2) * fsum( j -> e(j,k)*norm(d(j,k))^4)
#             - D/h * fsum( j -> e(j,k)*norm(d(j,k))^2)
#             + (D^2/2 - D) * fsum( j -> e(j,k) )
#             - 1/(2h^2) * fsum( j -> 1/(fsum(i->e(i,j))) * e(j,k) *  dot( d(j,k), fsum(i->e(i,j) * norm(d(i,j))^2 * d(i,j)) ) )
#             - 1/(2h^2) * fsum( j -> 1/(fsum(i->e(i,j))) * e(j,k) * norm(d(j,k))^2 * dot( d(j,k), fsum(i->e(i,j)*d(i,j)) ) )
#             + 1/(2h^2) * fsum( j -> (fsum(i->e(i,j)))^(-2) * fsum(i->e(i,j) * norm(d(i,j))^2) * e(j,k) * dot( d(j,k), fsum(i->e(i,j)*d(i,j)) ) )
#             + D/(2h) * fsum( j -> 1/(fsum(i->e(i,j))) * e(j,k) *  dot( d(j,k), fsum(i->e(i,j)*d(i,j)) ) )
#         )
#     end
#     fsum( k -> g(h,k)^2 / ( 2π^(D/2) ) )
# end
# export HE_loss_derivative
