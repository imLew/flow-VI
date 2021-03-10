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
    kernel.transform.s .= 1/sqrt(2*median_trick(q))
end

function median_cb!(kernel, q)
    kernel.transform.s .= 1/median(pairwise(Euclidean(), q, dims=2))
end

function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x, dims=2)
    # @info median(d)^2/log(size(x)[end])
    median(d)^2/log(size(x)[end])
end
