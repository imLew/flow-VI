using Plots
using Distributions
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind
true_color = colors[end-1]
therm_color = colors[end-2]
start_color = colors[end-3]

using Examples
const LogReg = LogisticRegression

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
export plot_classes
export plot_classes!
export plot_prediction
export plot_prediction!


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
    for q in data[:svgd_results]
        plot_2D_results!(plt, initial_dist, target_dist, q); 
    end
end

function plot_2D_gaussians_results(data; kwargs...)
    plt = plot(legend=false; kwargs...)
    plot_2D_gaussians_results!(plt, data)
    return plt
end

function make_boxplots(data::Array{Any}; kwargs...)
    kwargs = Dict(kwargs...)
    param_keys = intersect(keys(data[1]), [:α, :c₁, :c₂, :β₁, :β₂, :γ])
    if haskey(kwargs,:labels)
        labels = pop!(kwargs, :labels)
    else
        labels = [join(["$(key)=$(d[key])" for key in param_keys], "; ") 
                          for d in data] 
    end
    labels = reshape(labels, 1, length(data))
    plt = boxplot([d[:estimated_logZ] for d in data],
            labels=labels, colors=colors[1:length(data)],
            legend=:outerright; kwargs...) 
    if haskey(data[1], :true_logZ)
        hline!(plt, [data[1][:true_logZ]], label="true value", colors=true_color)
    end
    if haskey(data[1], :therm_logZ)
        hline!(plt, [data[1][:therm_logZ]], label="therm value", 
               colors=therm_color)
    end
    EV = expectation_V(data[1])
    H₀ = entropy(MvNormal(data[1][:μ₀], data[1][:Σ₀]))
    hline!(plt, [H₀ - EV], label="H₀ - E[V]", color=start_color)
    return plt
end

function plot_convergence(data, title=""; kwargs...)
    kwargs = Dict(kwargs...)
    legend = get!(kwargs, :legend, :bottomright)
    size = get!(kwargs, :size, (375, 375))
    ylims = get!(kwargs, :ylims, (-Inf, Inf))
    lw = get!(kwargs, :lw, 3)
    int_lims = get!(kwargs, :int_lims, (-Inf, Inf))

    int_plot, norm_plot = plot(), plot();
    results_plot = plot(legend=false);
    plot_convergence!(int_plot, results_plot, norm_plot, data; kwargs...)
    layout = @layout [ i ; n b ]
    return plot(int_plot, norm_plot, results_plot, layout=layout, title=title)
end

function plot_convergence!(int_plot, results_plot, norm_plot, data; kwargs...)
    kwargs = Dict(kwargs...)
    size = get!(kwargs, :size, (375, 375))
    legend = get!(kwargs, :legend, :bottomright)
    ylims = get!(kwargs, :ylims, (-Inf, Inf))
    lw = get!(kwargs, :lw, 3)
    int_lims = get!(kwargs, :int_lims, (-Inf, Inf))

    plot_integration!(int_plot, data, ylims=int_lims; kwargs...)

    if data[:problem_type] == :gauss_to_gauss
        plot_2D_gaussians_results!(results_plot, data)
    elseif data[:problem_type] == :logistic_regression
        plot_classes!(results_plot, data)
        plot_prediction!(results_plot, data)
    elseif data[:problem_type] == :linear_regression
        plot_fit!(results_plot, data)
    end
    
    if data[:n_runs] < 4
		for hist in data[:svgd_hist]
			plot!(norm_plot, hist[:ϕ_norm], ylims=ylims,
                  markeralpha=0, label="", title="", 
                  xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
                  xlabel="iterations", ylabel="||Δq||");
		end 
    else
        norms = [get(hist, :ϕ_norm)[2] for hist in data[:svgd_hist]]
        plot!(norm_plot, mean(norms), ribbon=std(norms), ylims=ylims,
              markeralpha=0, label="", title="", 
              xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
              xlabel="iterations", ylabel="||Δq||");
    end
end

function plot_integration(data; size=(375,375), legend=:bottomright, lw=3, 
                            ylims=(-Inf,Inf), title="", kwargs...)
    plt = plot(size=size, title=title);
    plot_integration!(plt, data; legend=legend, lw=lw, ylims=ylims)
end

function plot_integration!(plt::Plots.Plot, data; legend=:bottomright, 
                            lw=3, ylims=(-Inf,Inf), kwargs...)
    flow_label=get(kwargs, :flow_label, "")
    true_label=get(kwargs, :true_label, "")
    therm_label=get(kwargs, :therm_label, "")
    plot!(plt, xlabel="iterations", ylabel="log Z", legend=legend, lw=lw, 
          ylims=ylims);
    if data[:n_runs] < 5
        plot!(plt, estimate_logZ(data), color=colors[1], label=flow_label);
    else
        est_logZ = estimate_logZ(data)
        plot!(plt, mean(est_logZ), ribbon=std(est_logZ), color=colors[1], 
              label=flow_label);
    end
    if !isnothing(data[:true_logZ])
        hline!(plt, [data[:true_logZ]], labels=true_label, color=true_color, ls=:dash);
    end
    if !isnothing(data[:therm_logZ])
        hline!(plt, [data[:therm_logZ]], labels=therm_label, color=therm_color, ls=:dash);
    end
    EV = expectation_V(data)
    if haskey(data, :μ₀)
        H₀ = entropy(MvNormal(data[:μ₀], data[:Σ₀]))
    else
        H₀ = entropy(MvNormal(data[:μ_initial], data[:Σ_initial]))
    end
    hline!(plt, [H₀ - EV], label="H₀ - E[V]", color=start_color)
end

function plot_fit!(plt, data)
    x = range(data[:sample_range]..., length=100)
    for q in data[:svgd_results]
        for w in eachcol(q)
            model = LinReg.RegressionModel(data[:ϕ], w, data[:true_β])
            plot!(plt,x, LinReg.y(model), alpha=0.3, color=:orange, legend=:none)
        end
        plot!(plt, x, 
              LinReg.y(
               LinReg.RegressionModel(data[:ϕ], mean(q, dims=2), data[:true_β])
              ), 
              color=:red)
    end
    plot!(plt, x, 
          LinReg.y(
             LinReg.RegressionModel(data[:true_ϕ], data[:true_w], data[:true_β])
          ), 
          color=:green)
end

function plot_classes(data; kwargs...)
    plt = plot(;kwargs...);
    plot_classes!(Val(:logistic_regression), plt, data)
    return plt
end

function plot_classes(::Val{:logistic_regression}, data; kwargs...)
    plt = plot(;kwargs...);
    plot_classes!(Val(:logistic_regression), plt, data)
    return plt
end

function plot_classes!(plt, data)
    scatter!(plt, data[:D].x[:,1], data[:D].x[:,2], 
             legend=false, label="", colorbar=false, 
             zcolor=data[:D].t);
end

function plot_classes!(::Val{:logistic_regression}, plt, data)
    scatter!(plt, data[:D].x[:,1], data[:D].x[:,2], 
             legend=false, label="", colorbar=false, 
             zcolor=data[:D].t);
end

function plot_prediction(data)
    plt = plot()
    plot_prediction!(Val(data[:problem_type]), plt, data)
end

function plot_prediction!(plt, data)
    plot_prediction!(Val(data[:problem_type]), plt, data)
end

function plot_prediction!(::Val{:logistic_regression}, plt, data)
    xs = range(minimum(data[:D].x[:,1]), 
               maximum(data[:D].x[:,1]), length=100)
    ys = range(minimum(data[:D].x[:,2]), 
               maximum(data[:D].x[:,2]), length=100)
    grid = [[1, x, y] for x in xs, y in ys]

    σ(a) = 1 / (1 + exp(-a))
    q = hcat(data[:svgd_results]...)
    weights = [mean(d, dims=2)  for d in data[:svgd_results]]
    predictions = [σ(point'*w) for point in grid, w in eachcol(q)]
    avg_prediction = transpose(dropdims(mean(predictions, dims=3),dims=3))
    # heatmap() treats the y-direction as the first direction so the data needs
    # to be transposed before plotting it
    heatmap!(plt, xs, ys, avg_prediction, alpha=0.5);

    # for w in eachcol(q)
    for w in weights
        plot!(plt, x -> -(w[2]*x+w[1])/w[3], xs, legend=false, color=colors[1], 
              alpha=0.3, ylims=(minimum(ys), maximum(ys))
             );
    end
end

# export color_point_by_prediction!
# function color_point_by_prediction!(plt, data)
#     xs = range(minimum(data[:D][:,2]), maximum(data[:D][:,2]), length=100)
#     ys = range(minimum(data[:D][:,3]), maximum(data[:D][:,3]), length=100)
#     grid = [[1, x, y] for x in xs, y in ys]

#     σ(a) = 1 / (1 + exp(-a))
#     q = hcat(data[:svgd_results]...)
#     predictions = [σ(point'*w) for point in eachrow([ones(200) data[:D][:,2:end]]), w in eachcol(q)]
#     avg_prediction = mean(predictions, dims=3)
    
#     scatter!(plt, data[:D][:,2], data[:D][:,3], zcolor=avg_prediction)
# end

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
    append!(sp1.series_list, sp2.series_list)
    Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
    Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
end
