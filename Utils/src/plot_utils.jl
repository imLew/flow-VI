using Plots
using StatsPlots
using Distributions
using ColorSchemes
const colors = ColorSchemes.seaborn_colorblind
TRUE_COLOR = colors[end-1]
THERM_COLOR = colors[end-2]
START_COLOR = colors[end-3]
INT_COLOR = colors[end-4]

using Examples
LogReg = LogisticRegression
LinReg = LinearRegression

export plot_2D_results
export plot_2D_results!
export plot_1D
export plot_integration
export plot_integration!
export plot_convergence
export plot_convergence!
export plot_classes
export plot_classes!
export plot_prediction
export plot_prediction!
export make_boxplots
export make_boxplots!
export plot_annealing_schedule
export plot_annealing_schedule!

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

function _get_dist_bounds(initial_dist::MvNormal, target_dist::MvNormal)
    initial_x = params(initial_dist)[2][1,1]
    initial_y = params(initial_dist)[2][2,2]
    target_x = params(target_dist)[2][1,1]
    target_y = params(target_dist)[2][2,2]
    max_target_y = mean(target_dist)[2] + 2*target_y
    min_target_y = mean(target_dist)[2] - 2*target_y
    max_target_x = mean(target_dist)[1] + 2*target_x
    min_target_x = mean(target_dist)[1] - 2*target_x
    return (initial_x, initial_y, min_target_x, min_target_y, max_target_x,
            max_target_y)
end

function _get_dist_bounds(
    initial_dist::MvNormal,
    target_dist::MixtureModel{Multivariate, Continuous, MvNormal}
)
    initial_x = params(initial_dist)[2][1,1]
    initial_y = params(initial_dist)[2][2,2]
    target_covariances = [x[2] for x in params(target_dist)[1]]
    target_means = [x[1] for x in Distributions.params(target_dist)[1]]
    target_x = maximum([Σ[1,1] for Σ in target_covariances])
    target_y = maximum([Σ[2,2] for Σ in target_covariances])
    min_target_y = minimum([μ[2] for μ in target_means]) - 2*target_y
    max_target_y = maximum([μ[2] for μ in target_means]) + 2*target_y
    min_target_x = minimum([μ[1] for μ in target_means]) - 2*target_x
    max_target_x = maximum([μ[1] for μ in target_means]) + 2*target_x
    return (initial_x, initial_y, min_target_x, min_target_y, max_target_x,
            max_target_y)
end

function _get_scatter_range(initial_dist, target_dist, q)
    (initial_x, initial_y, min_target_x, min_target_y, max_target_x,
     max_target_y) = _get_dist_bounds(initial_dist, target_dist)
    min_x = minimum([
                     minimum(q[1]) - 0.1 * abs(minimum(q[1])),
                     mean(initial_dist)[1] - initial_x,
                     min_target_x
                    ])
    max_x = maximum([
                     maximum(q[1]) + 0.1 * abs(maximum(q[1])),
                     mean(initial_dist)[1] + initial_x,
                     max_target_x
                    ])
    min_y = minimum([
                     minimum(q[2]) - 0.1 * abs(minimum(q[2])),
                     mean(initial_dist)[2] - initial_y,
                     min_target_y
                    ])
    max_y = maximum([
                     maximum(q[2]) + 0.1 * abs(maximum(q[2])),
                     mean(initial_dist)[2] + initial_y,
                     max_target_y
                    ])
    x = min_x:abs(max_x-min_x)/50:max_x
    y = min_y:abs(max_y-min_y)/50:max_y
    return x, y
end

function plot_2D_results!(
    plt,
    initial_dist::Distribution,
    target_dist::MixtureModel{Multivariate, Continuous, MvNormal},
    q
)
    # get range to cover both distributions and the particles
    f(x,y) = pdf(target_dist, [x, y]) #/ pdf(target_dist, [mean(target_dist)...])
    g(x,y) = pdf(initial_dist, [x, y]) #/ pdf(initial_dist, [mean(initial_dist)...])
    x, y = _get_scatter_range(initial_dist, target_dist, q)
    heatmap!(plt, x, y, f,label="", levels=50,
             color=cgrad([:white, colors[2], colors[3]]),
             markerstrokewidths=0.0, alpha=0.6, )
    contour!(plt, x, y, g, color=colors[1], label="", levels=5,
             markerstrokewidths=0.0, alpha=0.6, )
    scatter!(plt, q[1,:], q[2,:], legend=false, label="",
             markerstrokewidths=1.0, alpha=1, color=colors[1],
             markersize=1);
    return plt
end

function plot_2D_results!(
    plt,
    initial_dist::Distribution,
    target_dist::MvNormal,
    q
)
    # get range to cover both distributions and the particles
    f(x,y) = pdf(target_dist, [x, y]) / pdf(target_dist, [mean(target_dist)...])
    g(x,y) = pdf(initial_dist, [x, y]) / pdf(initial_dist, [mean(initial_dist)...])
    x, y = _get_scatter_range(initial_dist, target_dist, q)
    contour!(plt, x, y, f, color=colors[2], label="", levels=5,
             markerstrokewidths=0.0, alpha=0.6, )
    contour!(plt, x, y, g, color=colors[1], label="", levels=5,
             markerstrokewidths=0.0, alpha=0.6, )
    scatter!(plt, q[1,:], q[2,:], legend=false, label="",
             markerstrokewidths=0.0, alpha=0.5, color=colors[1],
             markersize=1);
    return plt
end

function plot_2D_results(
    initial_dist::Distribution,
    target_dist::Distribution,
    q,
    ;kwargs...
)
    plt = plot(legend=false; kwargs...)
    plot_2D_results!(plt, initial_dist, target_dist, q)
    return plt
end

function plot_2D_results!(plt, data)
    if data[:problem_type] == :gauss_to_gauss
        initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
        target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    elseif data[:problem_type] == :linear_regression
        initial_dist = MvNormal(data[:μ_initial], data[:Σ_initial])
        Σ = LinReg.posterior_variance(data[:true_ϕ], data[:true_β],
                                      data[:D].x, data[:Σ_prior])
        μ = LinReg.posterior_mean(data[:true_ϕ], data[:true_β], data[:D],
                                      data[:μ_prior], data[:Σ_prior])
        target_dist = MvNormal(μ, Σ)
    elseif data[:problem_type] == :gauss_mixture_sampling
        initial_dist = MvNormal(data[:μ_initial], data[:Σ_initial])
        target_dist = MixtureModel(MvNormal, [zip(data[:μₚ], data[:Σₚ])...])
    end
    for q in data[:svgd_results]
        plot_2D_results!(plt, initial_dist, target_dist, q);
    end
end

function plot_2D_results(data; kwargs...)
    plt = plot(legend=false; kwargs...)
    plot_2D_results!(plt, data)
    return plt
end

function make_legend(labels; kwargs...)
    colors = get!(kwargs, :colors, colors)
    styles = get!(kwargs, :styles, [:line for l in labels])

    # rkhss = :dot
    # steins = :dash
    # usteins = :dashdot
    # truths = :solid
    plt = plot([])
    for (l, c, s) in zip(labels, colors, styles)
        plot!(plt, [], label=l, color=c, style=s,
              lw = 20.0, showaxis = false, framestyle=:none, legend = :left,
              legendfontsize = 30.0, foreground_color_legend = RGBA(0,0,0,0),
              background_color_legend = RGBA(0,0,0,0)
             )
    end

    # p = plot([[],[],[],[]], [[],[],[],[]],
    #          color = [rkhsc steinc usteinc truthc],
    #          linestyle = [rkhss steins usteins truths],
    #          label = ["RKHS" "Stein" "Unbiased Stein" "Truth"],
    #          lw = 20.0,
    #          showaxis = false,
    #          framestyle=:none,
    #          legend = :left,
    #          legendfontsize = 30.0,
    #          foreground_color_legend = RGBA(0,0,0,0),
    #          background_color_legend = RGBA(0,0,0,0),
    #         )
end

function make_boxplots(data; legend_keys=[], kwargs...)
    plt = plot()
    make_boxplots!(plt, data, legend_keys=legend_keys; kwargs...)
end

function make_boxplots!(plt, data; legend_keys=[], kwargs...)
    kwargs = Dict(kwargs...)
    true_label=get(kwargs, :true_label, "")
    therm_label=get(kwargs, :therm_label, "")
    start_label=get(kwargs, :start_label, "")
    int_color = get(kwargs, :int_color, INT_COLOR)
    true_color = get(kwargs, :true_color, TRUE_COLOR)
    therm_color = get(kwargs, :therm_color, THERM_COLOR)
    start_color = get(kwargs, :start_color, START_COLOR)
    box_colors = get(kwargs, :box_colors, colors)
    legend_keys = intersect(keys(data[1]), legend_keys)
    !haskey(kwargs, :xticks) ? kwargs[:xticks] = [] : nothing
    if haskey(kwargs,:labels)
        labels = pop!(kwargs, :labels)
    else
        labels = [join(["$(key)=$(d[key])" for key in legend_keys], "; ")
                          for d in data]
    end
    labels = reshape(labels, 1, length(data))
    boxplot!(plt,
            [[est[end] for est in estimate_logZ(d; kwargs...)] for d in data],
            labels=labels; kwargs...
           )
    if haskey(data[1], :true_logZ)
        hline!(plt, [data[1][:true_logZ]], label=true_label, color=true_color,
              lw=2)
    end
    if haskey(data[1], :therm_logZ) && !isnothing(data[1][:therm_logZ])
        hline!(plt, [data[1][:therm_logZ]], label=therm_label, ls=:dot,
               colors=therm_color, lw=2)
    end
    if haskey(data[1], :EV)
        EV = data[1][:EV]
    else
        EV = expectation_V(data[1]; kwargs...)
    end
    if data[1][:problem_type] == :gauss_to_gauss
        H₀ = entropy(MvNormal(data[1][:μ₀], data[1][:Σ₀]))
    else
        H₀ = entropy(MvNormal(data[1][:μ_initial], data[1][:Σ_initial]))
    end
    hline!(plt, [H₀ - EV], label=start_label, color=start_color, ls=:dot, lw=2)
    return plt
end

function plot_convergence(data; title="", kwargs...)
    int_plot, norm_plot = plot(title=title), plot();
    results_plot = plot(legend=false);
    plot_convergence!(int_plot, results_plot, norm_plot, data; kwargs...)
    if data[:problem_type]==:linear_regression && length(data[:μ_initial])==2
        gamma_plot = plot(get(data[:svgd_hist][1], :annealing)[2]);
        dist_plot = plot_2D_results(data)
        layout = @layout [ i ; n g ; f d ]
        return plot(int_plot, norm_plot, gamma_plot, results_plot, dist_plot,
                    layout=layout)
    else
        layout = @layout [ i ; n b ]
        return plot(int_plot, norm_plot, results_plot, layout=layout; kwargs...)
    end
end

function plot_convergence!(
    int_plot,
    results_plot,
    norm_plot,
    data,
    ;kwargs...
)
    kwargs = Dict(kwargs...)
    size = get(kwargs, :size, (375, 375))
    legend = get(kwargs, :legend, :bottomright)
    ylims = get(kwargs, :ylims, (0, Inf))
    xlims = get(kwargs, :xlims, (0, Inf))
    lw = get(kwargs, :lw, 3)
    int_lims = get(kwargs, :int_lims, (-Inf, Inf))

    plot_integration!(int_plot, data, xlims=xlims, ylims=int_lims; kwargs...)

    if data[:problem_type] in [:gauss_to_gauss, :gauss_mixture_sampling]
        plot_2D_results!(results_plot, data)
    elseif data[:problem_type] == :logistic_regression
        plot_classes!(results_plot, data)
        plot_prediction!(results_plot, data)
    elseif data[:problem_type] == :linear_regression
        plot_fit!(results_plot, data)
    end

    if data[:n_runs] < 4
		for hist in data[:svgd_hist]
			plot!(norm_plot, hist[:ϕ_norm], ylims=ylims,
                  markeralpha=0, label="", title="", xlims=xlims,
                  xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
                  xlabel="iterations", ylabel="||Δq||");
		end
    else
        norms = [get(hist, :ϕ_norm)[2] for hist in data[:svgd_hist]]
        plot!(norm_plot, mean(norms), ribbon=std(norms), ylims=ylims,
              markeralpha=0, label="", title="", xlims=xlims,
              xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
              xlabel="iterations", ylabel="||Δq||");
    end
end

function plot_integration(
    data,
    ;size=(375,375),
    legend=:bottomright,
    lw=3,
    ylims=(-Inf,Inf),
    title="",
    kwargs...
)
    plt = plot(size=size, title=title; kwargs...);
    plot_integration!(plt, data, legend=legend, lw=lw, ylims=ylims; kwargs...)
end

function plot_integration!(
    plt::Plots.Plot, data,
    ; legend=:bottomright,
    lw=3,
    ylims=(-Inf,Inf),
    kwargs...
)
    flow_label = get(kwargs, :flow_label, "")
    true_label = get(kwargs, :true_label, "")
    therm_label = get(kwargs, :therm_label, "")
    start_label = get(kwargs, :start_label, "")
    therm_color = get(kwargs, :therm_color, THERM_COLOR)
    int_color = get(kwargs, :int_color, INT_COLOR)
    true_color = get(kwargs, :true_color, TRUE_COLOR)
    start_color = get(kwargs, :start_color, START_COLOR)
    show_ribbon = get(kwargs, :show_ribbon, true)
    plot!(plt, xlabel="iterations", ylabel="log Z", legend=legend, lw=lw,
          ylims=ylims);
    est_logZ = estimate_logZ(data; kwargs...)
    if typeof(est_logZ) <: Dict{Any, Any}
        for ((estimator, estimate), ls) in zip(est_logZ, [:solid, :dot, :dash])
            if data[:n_runs] < 5
                plot!(plt, estimate, color=int_color,
                      label=flow_label*" $estimator", ls=ls);
            else
                if show_ribbon
                    plot!(plt, mean(estimate), ribbon=std(estimate),
                          color=int_color, label=flow_label, ls=ls);
                else
                    plot!(plt, mean(estimate), color=int_color,
                          label=flow_label, ls=ls);
                end
            end
        end
    else
        if data[:n_runs] < 5
            plot!(plt, est_logZ, color=int_color, label=flow_label);
        else
            if show_ribbon
                plot!(plt, mean(est_logZ), ribbon=std(est_logZ),
                      color=int_color, label=flow_label);
            else
                plot!(plt, mean(est_logZ), color=int_color,
                      label=flow_label);
            end
        end
    end
    if haskey(data, :true_logZ)
        hline!(plt, [data[:true_logZ]], labels=true_label, color=true_color);
    end
    if haskey(data, :therm_logZ) && !isnothing(data[:therm_logZ])
        hline!(plt, [data[:therm_logZ]], labels=therm_label, color=therm_color,
               ls=:dashdot);
    end
    if haskey(data, :EV)
        EV = data[:EV]
    else
        EV = expectation_V(data; kwargs...)
    end
    if data[:problem_type] == :gauss_to_gauss
        H₀ = entropy(MvNormal(data[:μ₀], data[:Σ₀]))
    else
        H₀ = entropy(MvNormal(data[:μ_initial], data[:Σ_initial]))
    end
    hline!(plt, [H₀ - EV], label=start_label, color=start_color, ls=:dot)
end

function plot_fit!(plt, data)
    x = range(data[:sample_range]..., length=100)
    for q in data[:svgd_results]
        for w in eachcol(q)
            model = LinReg.RegressionModel(data[:ϕ], w, data[:true_β])
            plot!(plt,x, x -> LinReg.y(model, x), alpha=0.3, color=:orange,
                  legend=:none)
        end
        plot!(plt, x,
              x -> LinReg.y(
               LinReg.RegressionModel(data[:ϕ], mean(q, dims=2), data[:true_β]),
               x
              ),
              color=:red)
    end
    plot!(plt, x,
          x -> LinReg.y(
             LinReg.RegressionModel(data[:true_ϕ], data[:true_w], data[:true_β]),
             x
          ),
          color=:green)
end

function plot_classes(data; kwargs...)
    plt = plot(;kwargs...);
    plot_classes!(Val(:logistic_regression), plt, data; kwargs...)
    return plt
end

function plot_classes(::Val{:logistic_regression}, data; kwargs...)
    plt = plot(;kwargs...);
    plot_classes!(Val(:logistic_regression), plt, data; kwargs...)
    return plt
end

function plot_classes!(plt, D::LogReg.Data; kwargs...)
    scatter!(plt, D.x[:,1], D.x[:,2],
             legend=false, label="", colorbar=false,
             zcolor=D.t; kwargs...);
end

function plot_classes!(plt, data; kwargs...)
    # scatter!(plt, data[:D].x[:,1], data[:D].x[:,2],
    #          legend=false, label="", colorbar=false,
    #          zcolor=data[:D].t; kwargs...);
    plot_classes!(plt, data[:D]; kwargs...)
end

function plot_classes!(::Val{:logistic_regression}, plt, data; kwargs...)
    # scatter!(plt, data[:D].x[:,1], data[:D].x[:,2],
    #          legend=false, label="", colorbar=false,
    #          zcolor=data[:D].t);
    plot_classes!(plt, data[:D]; kwargs...)
end

function plot_prediction(data)
    plt = plot()
    plot_prediction!(Val(data[:problem_type]), plt, data)
end

function plot_prediction!(plt, data)
    plot_prediction!(Val(data[:problem_type]), plt, data)
end

function plot_prediction!(::Val{:logistic_regression}, plt, data)
    plot_prediction!(plt, data[:D], data[:svgd_results])
end

function plot_prediction!(plt, D, q)
    xs = range(minimum(D.x[:,1]),
               maximum(D.x[:,1]), length=100)
    ys = range(minimum(D.x[:,2]),
               maximum(D.x[:,2]), length=100)
    grid = [[1, x, y] for x in xs, y in ys]

    σ(a) = 1 / (1 + exp(-a))
    weights = [mean(d, dims=2)  for d in q]
    q = hcat(q...)
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

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
    append!(sp1.series_list, sp2.series_list)
    Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
    Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
end

function plot_annealing_schedule!(plt, data::Dict{Symbol, Any}; kwargs...)
    γ = get(data[:svgd_hist][1], :annealing)[2]
    plot!(plt, γ; kwargs...)
end

function plot_annealing_schedule(data::Dict{Symbol, Any}; kwargs...)
    plt = plot(;kwargs...)
    plot_annealing_schedule!(plt, data; kwargs...)
    return plt
end

function plot_annealing_schedule!(plt, data; kwargs...)
    for d in data
        if haskey(d, :annealing_schedule)
            plot_annealing_schedule!(plt, d; kwargs...)
        end
    end
    return plt
end

function plot_annealing_schedule(data; kwargs...)
    plt = plot(;kwargs...)
    for d in data
        if haskey(d, :annealing_schedule)
            plot_annealing_schedule!(plt, d; kwargs...)
        end
    end
    return plt
end
