export linear_annealing
export hyperbolic_annealing
export cyclic_annealing

function annealing(::Val{:cyclic}, iter, max_iter; kwargs...)
    @unpack p, C = kwargs
    return ( (mod(iter, max_iter/C)+1)/(max_iter/C) )^p
end

function annealing(::Val{:hyperbolic}, iter, max_iter; kwargs...)
    @unpack p = kwargs
    return tanh( (1.3 * (iter/max_iter))^p )
end

function annealing(::Val{:linear}, iter, max_iter; kwargs...)
    return iter/max_iter
end

function annealing(iter, max_iter, duration, type; kwargs...)
    if iter/max_iter < duration
        T = round(duration * max_iter)
        return annealing(Val(type), iter, T; kwargs...)
    else
        return 1.
    end
end

function linear_annealing(iter, max_iter; duration=0.8, kwargs...)
    annealing(iter, max_iter, duration, :linear; kwargs...)
end

function hyperbolic_annealing(iter, max_iter; duration=0.8, p=1, kwargs...)
    annealing(iter, max_iter, duration, :hyperbolic, p=p)
end

function cyclic_annealing(iter, max_iter; duration=0.8, p=1, C=5, kwargs...)
    annealing(iter, max_iter, duration, :cyclic, p=p, C=C)
end
