using Plots
using Distributions
using ColorSchemes

const colors = ColorSchemes.seaborn_colorblind

# export plot_known_dists
export plot_2D_results
export plot_2D_results!
export plot_2D_gaussians_results
export plot_2D_gaussians_results!
export plot_1D
export plot_integration
export plot_integration!
export plot_convergence
export plot_convergence!

function plot_1D(initial_dist::Distribution, target_dist::Distribution, q)
    n_bins = length(q) ÷ 5
    dist_plot = histogram(reshape(q, length(q)), 
                          fillalpha=0.3, labels="q" ,bins=20,
                          normalize=true);
    min_x = minimum([ 
                     minimum(q) - 0.2 * abs(minimum(q)), 
                     mean(initial_dist) - 3*norm(cov(initial_dist)), 
                     mean(target_dist) - 3*norm(cov(target_dist))
                   ])
    max_x = maximum([ 
                     maximum(q) + 0.2 * abs(maximum(q)), 
                     mean(initial_dist) + 3*norm(cov(initial_dist)), 
                     mean(target_dist) + 3*norm(cov(target_dist))
                   ])
    t = min_x-0.2*abs(min_x):0.05:max_x+0.2*abs(max_x)
    plot!(x->pdf(target_dist, x), t, labels="p")
    plot!(x->pdf(initial_dist, x), t, labels="q₀")
    return dist_plot
end

function plot_2D_results!(plt, initial_dist::Distribution, 
                          target_dist::Distribution, q)
    scatter!(plt, q[1,:], q[2,:], legend=false, label="", msw=0.0, alpha=0.5, color=colors[1]);
    # get range to cover both distributions and the particles
    min_x = minimum([
                     minimum(q[1]) - 0.2 * abs(minimum(q[1])), 
                     mean(initial_dist)[1] - 3*params(initial_dist)[2][1,1], 
                     mean(target_dist)[1] - 3*params(target_dist)[2][1,1]
                   ])
    max_x = maximum([
                     maximum(q[1]) + 0.2 * abs(maximum(q[1])), 
                    mean(initial_dist)[1] + 3*params(initial_dist)[2][1,1], 
                    mean(target_dist)[1] + 3*params(target_dist)[2][1,1]
                   ])
    min_y = minimum([
                     minimum(q[2]) - 0.2 * abs(minimum(q[2])), 
                    mean(initial_dist)[2] - 3*params(initial_dist)[2][2,2], 
                    mean(target_dist)[2] - 3*params(target_dist)[2][2,2]
                   ])
    max_y = maximum([
                     maximum(q[2]) + 0.2 * abs(maximum(q[2])), 
                    mean(initial_dist)[2] + 3*params(initial_dist)[2][2,2], 
                    mean(target_dist)[2] + 3*params(target_dist)[2][2,2]
                   ])
    x = min_x:0.05:max_x
    y = min_y:0.05:max_y
    contour!(plt, x, y, (x,y)->pdf(target_dist, [x, y]), color=colors[2], 
             label="", levels=5, msw=0.0, alpha=0.6)
    contour!(plt, x, y, (x,y)->pdf(initial_dist, [x, y]), color=colors[1], 
             label="", levels=5, msw=0.0, alpha=0.6)
    return plt
end

function plot_2D_results(initial_dist::Distribution, target_dist::Distribution, 
                         q; kwargs...)
    plt = plot(legend=false; kwargs...)
    plot_2D_results!(plt, initial_dist, target_dist, q)
    return plt
end

function plot_2D_gaussians_results!(plt, data)
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    plot_2D_results!(plt, initial_dist, target_dist, data[:svgd_results][1][end]); 
end

function plot_2D_gaussians_results(data; kwargs...)
    plt = plot(legend=false; kwargs...)
    plot_2D_gaussians_results!(plt, data)
    return plt
end

function plot_convergence(data; kwargs...)
    kwargs = Dict(kwargs...)
    legend = get!(kwargs, :legend, :bottomright)
    size = get!(kwargs, :size, (375, 375))
    ylims = get!(kwargs, :ylims, (-Inf, Inf))
    lw = get!(kwargs, :lw, 3)
    int_lims = get!(kwargs, :int_lims, (-Inf, Inf))

    plt, int_plot, norm_plot = plot(), plot(), plot()
    dist_plot = plot(legend=false) 
    plot_convergence!(int_plot, dist_plot, norm_plot, data; kwargs...)
    layout = @layout [ i ; n b ]
    plot(int_plot, norm_plot, dist_plot, layout=layout)
end

function plot_convergence!(int_plot, dist_plot, norm_plot, data; kwargs...)
    kwargs = Dict(kwargs...)
    size = get!(kwargs, :size, (375, 375))
    legend = get!(kwargs, :legend, :bottomright)
    ylims = get!(kwargs, :ylims, (-Inf, Inf))
    lw = get!(kwargs, :lw, 3)
    int_lims = get!(kwargs, :int_lims, (-Inf, Inf))

    plot_integration!(int_plot, data)

    plot_2D_gaussians_results!(dist_plot, data)
    
    plot!(norm_plot, data[:svgd_results][1][1][:ϕ_norm],ylims=(0,Inf),
                     markeralpha=0, label="", title="", 
                     xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
                     xlabel="iterations", ylabel="||φ||");
end

function plot_integration(data; size=(375,375), legend=:bottomright, lw=3, 
                          ylims=(-Inf,Inf))
    plt = plot()
    plot_integration!(plt, data; size=size, legend=legend, lw=lw, ylims=ylims)
end

function plot_integration!(plt::Plots.Plot, data; size=(375,375),
                           legend=:bottomright, lw=3, ylims=(-Inf,Inf))
    dKL_hist = data[:svgd_results][1][1]
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)
    plot!(plt, xlabel="iterations", ylabel="log Z", legend=legend, lw=lw, ylims=ylims);
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :RKHS_norm)[2]))
    plot!(plt, est_logZ, label="", color=colors[1]);
    hline!(plt, [true_logZ], labels="", color=colors[2], ls=:dash);
end

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
    append!(sp1.series_list, sp2.series_list)
    Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
    Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
end

export plot_classes
export plot_classes!

function plot_classes(sample_data)
    plt = plot()
    plot_classes!(plt, sample_data)
    return plt
end

function plot_classes!(plt, sample_data)
    scatter!(plt, sample_data[:,2], sample_data[:,3], legend=false, label="", 
            colorbar=false, msw=0.0, alpha=0.5, zcolor=sample_data[:,1]);
end

function plot_prediction!(plt, data)
    xs = range(minimum(data[:sample_data][:,2]), maximum(data[:sample_data][:,2]), length=100)
    ys = range(minimum(data[:sample_data][:,3]), maximum(data[:sample_data][:,3]), length=100)
    grid = [[1, x, y] for x in xs, y in ys]

    σ(a) = 1 / (1 + exp(-a))
    q = hcat(data[:svgd_results]...)
    predictions = [σ(point'*w) for point in grid, w in eachcol(q)]
    avg_prediction = transpose(dropdims(mean(predictions, dims=3),dims=3))
    # for now leave the transpose as it seems to give the correct result    
    heatmap!(plt, xs, ys, avg_prediction, alpha=0.5)

    for w in eachcol(q)
        plot!(plt, x -> -(w[2]*x+w[1])/w[3], xs, legend=false, color=colors[1], alpha=0.3)
    end
    display(plt)
end

function color_point_by_prediction!(plt, data)
    xs = range(minimum(data[:sample_data][:,2]), maximum(data[:sample_data][:,2]), length=100)
    ys = range(minimum(data[:sample_data][:,3]), maximum(data[:sample_data][:,3]), length=100)
    grid = [[1, x, y] for x in xs, y in ys]

    σ(a) = 1 / (1 + exp(-a))
    q = hcat(data[:svgd_results]...)
    # predictions = [σ(point'*w) for point in grid, w in eachcol(q)]
    predictions = [σ(point'*w) for point in eachrow([ones(200) data[:sample_data][:,2:end]]), w in eachcol(q)]
    avg_prediction = mean(predictions, dims=3)
    
    scatter!(plt, data[:sample_data][:,2], data[:sample_data][:,3], zcolor=avg_prediction)
    # heatmap!(plt, xs, ys, dropdims(avg_prediction,dims=3), alpha=0.5)
end

export plot_prediction!
export color_point_by_prediction!

