using Plots
using Distributions
using LinearAlgebra
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind

using Utils
using Examples
const LogReg = LogisticRegression

export plot_classes
export plot_classes!
export plot_prediction!
export plot_integration
export plot_integration!

function plot_integration(data; size=(375,375), legend=:bottomright, lw=3, 
                          ylims=(-Inf,Inf))
    plt = plot()
    plot_integration!(plt, data; size=size, legend=legend, lw=lw, ylims=ylims)
end

function plot_integration!(plt::Plots.Plot, data; size=(375,375),
                           legend=:bottomright, lw=3, ylims=(-Inf,Inf))
    # dKL_hist = data[:svgd_results][1]
    initial_dist = MvNormal(data[:μ_initial], data[:Σ_initial])
    H₀ = Distributions.entropy(initial_dist)
    EV = ( num_expectation( initial_dist, 
                                  w -> LogReg.log_likelihood(data[:sample_data],w),
                                 )
           + expectation_V(initial_dist, initial_dist) 
           + 0.5 * logdet(2π * data[:Σ_initial]) 
          )
    true_logZ = data[:therm_logZ]
    plot!(plt, xlabel="iterations", ylabel="log Z", legend=legend, lw=lw, ylims=ylims);
    if data[:n_runs] < 5
        for dKL_hist in data[:svgd_hist]
        est_logZ = estimate_logZ.([H₀], [EV], 
                                  data[:step_size]*cumsum(get(dKL_hist, :RKHS_norm)[2]))
        plot!(plt, est_logZ, label="", color=colors[1]);
        end
    else
        est_logZ = [estimate_logZ.([H₀], [EV], 
                                  data[:step_size]*cumsum(get(d, :RKHS_norm)[2]))
                    for d in data[:svgd_hist]]
        plot!(plt, mean(est_logZ), ribbon=std(est_logZ), label="", color=colors[1]);
    end
    hline!(plt, [true_logZ], labels="", color=colors[2], ls=:dash);
end

function plot_classes(sample_data; kwargs...)
    plt = plot(;kwargs...)
    plot_classes!(plt, sample_data)
    return plt
end

function plot_classes!(plt, sample_data)
    scatter!(plt, sample_data.x[:,1], sample_data.x[:,2], legend=false, label="", 
            colorbar=false, zcolor=sample_data.t);
end

function plot_prediction!(plt, data)
    xs = range(minimum(data[:sample_data].x[:,1]), maximum(data[:sample_data].x[:,1]), length=100)
    ys = range(minimum(data[:sample_data].x[:,2]), maximum(data[:sample_data].x[:,2]), length=100)
    grid = [[1, x, y] for x in xs, y in ys]

    σ(a) = 1 / (1 + exp(-a))
    q = hcat(data[:svgd_results]...)
    weights = [mean(d, dims=2)  for d in data[:svgd_results]]
    predictions = [σ(point'*w) for point in grid, w in eachcol(q)]
    avg_prediction = transpose(dropdims(mean(predictions, dims=3),dims=3))
    # heatmap() treats the y-direction as the first direction so the data needs
    # to be transposed before plotting it
    heatmap!(plt, xs, ys, avg_prediction, alpha=0.5)

    # for w in eachcol(q)
    for w in weights
        plot!(plt, x -> -(w[2]*x+w[1])/w[3], xs, legend=false, color=colors[1], 
              alpha=0.3, ylims=(minimum(ys), maximum(ys))
             )
    end
    display(plt)
end

# export color_point_by_prediction!
# function color_point_by_prediction!(plt, data)
#     xs = range(minimum(data[:sample_data][:,2]), maximum(data[:sample_data][:,2]), length=100)
#     ys = range(minimum(data[:sample_data][:,3]), maximum(data[:sample_data][:,3]), length=100)
#     grid = [[1, x, y] for x in xs, y in ys]

#     σ(a) = 1 / (1 + exp(-a))
#     q = hcat(data[:svgd_results]...)
#     predictions = [σ(point'*w) for point in eachrow([ones(200) data[:sample_data][:,2:end]]), w in eachcol(q)]
#     avg_prediction = mean(predictions, dims=3)
    
#     scatter!(plt, data[:sample_data][:,2], data[:sample_data][:,3], zcolor=avg_prediction)
# end
