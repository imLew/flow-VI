using DrWatson
using BSON
using OnlineStats
using Distributions
using ValueHistories
using Optim
using LinearAlgebra
using ProgressMeter

using Utils
using Examples
LogReg = LogisticRegression

###### Cell ###### -
all_data = load_data(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_50"))
d = all_data[2]

###### Cell ###### -
for d in all_data
    H₀ = entropy(MvNormal(collect(d[:μ_initial]), collect(d[:Σ_initial])))
    𝔼V = expectation_V(d, n_samples=10000)
    @info H₀ - 𝔼V
end
# Info: -13.523540152208176  # Laplace
# Info: -44.646442064624125  # MAP

###### Cell ###### -

###### Cell ###### -
𝓁(x) = LogReg.likelihood(all_data[1][:D], x)  # same for all experiments
prior(x) = pdf(MvNormal(collect(d[:μ_prior]), collect(d[:Σ_prior])), x)
joint(x) = 𝓁(x)*prior(x)

###### Cell ###### -
# TI_logZ = therm_integration(d, nSteps=50)

d[:estimated_logZ]
for d in all_data
    @info mean(d[:estimated_logZ])
end

# as importance distribution use gaussian centered on maximum likelihood
###### Cell ###### -

∇𝓁!(g, w) = g .= vec(LogReg.grad_log_likelihood(d[:D], w))
maxL = Optim.maximizer( Optim.maximize(𝓁, ∇𝓁!, d[:μ_prior], LBFGS()) )

q_dist = MvNormal(maxL, collect(I(3)))
q(x) = pdf(q_dist, x)

Z = Mean()

N = Int(1e5)
Threads.@threads for i in 1:100
    fit!(Z, map(x->joint(x)/q(x), eachcol(rand(q_dist, N))) )
    @info log(value(Z))
end

# as importance distribution use mixture of gausssians centered on
# maximum likelihood and prior
###### Cell ###### -

qM_dist = MixtureModel(MvNormal, [(maxL, I(3)), (d[:μ_prior], d[:Σ_prior])])
qM(x) = pdf(qM_dist, x)

ZM = Mean()

N = Int(1e5)
Threads.@threads for i in 1:100
    fit!(ZM, map(x->joint(x)/qM(x), eachcol(rand(qM_dist, N))) )
    @info log(value(ZM))
end

###### Cell ###### -
# as importance distribution use Laplace approximation to posterior

qL_dist = MvNormal(d[:μ_initial], d[:Σ_initial])
qL(x) = pdf(qL_dist, x)

ZL = Mean()

N = Int(1e5)
Threads.@threads for i in 1:100
    fit!(ZL, map(x->joint(x)/qL(x), eachcol(rand(qL_dist, N))) )
    @info log(value(ZL))
end

###### Cell ###### -
# as importance distribution use Laplace approximation to posterior with
# much wider variance

qwL_dist = MvNormal(d[:μ_initial], 100.0*d[:Σ_initial])
qwL(x) = pdf(qwL_dist, x)

ZwL = Mean()

N = Int(1e5)
Threads.@threads for i in 1:100
    fit!(ZwL, map(x->joint(x)/qwL(x), eachcol(rand(qwL_dist, N))) )
    @info log(value(ZwL))
end

###### Cell ###### -
qwLM_dist = MixtureModel(MvNormal, [(d[:μ_initial], 100.0*d[:Σ_initial]),
                                   (d[:μ_prior], d[:Σ_prior])])
qwLM(x) = pdf(qwLM_dist, x)

ZwLM = Mean()

N = Int(1e5)
Threads.@threads for i in 1:100
    fit!(ZwLM, map(x->joint(x)/qwLM(x), eachcol(rand(qwLM_dist, N))) )
    @info log(value(ZwLM))
end

###### Cell ###### - mixture of MAP, ML and prior
qA_dist = MixtureModel(MvNormal, [(maxL, I(3)),
                                  (d[:μ_initial], d[:Σ_initial]),
                                   (d[:μ_prior], d[:Σ_prior])])
qA(x) = pdf(qA_dist, x)

ZA = Mean()

N = Int(1e5)
p = Progress(1000)
Threads.@threads for i in 1:1000
    fit!(ZA, map(x->joint(x)/qA(x), eachcol(rand(qA_dist, N))) )
    next!(p)
end

###### Cell ###### -
for z in [Z, ZM, ZL, ZwL, ZwLM]
    @info log(value(z)), z.n
end

x = [log(value(z)) for z in [ZM, ZL, ZwL, ZwLM]]
push!(x, -13.340725856146562)  # This is the value of ZA, which was computed in a separate instance for speed.
true_logZ = mean(x)  # -13.3395063723414

###### Cell ###### - add true log estimate

for (i,n) in enumerate(readdir(gdatadir("bayesian_logistic_regression", "MAPvLaplace_rerun_50"), join=true))
    e = merge(all_data[i], @dict(true_logZ))
    bson(n, e)
end

for dir in ["MAPvLaplace_rerun_10", "MAPvLaplace_rerun_25", "MAPvLaplace_rerun_100"]
    ad = load_data(gdatadir("bayesian_logistic_regression",dir))
    for (i,n) in enumerate(readdir(gdatadir("bayesian_logistic_regression", dir), join=true))
        e = merge(ad[i], @dict(true_logZ))
        bson(n, e)
    end
end


###### Cell ###### -
