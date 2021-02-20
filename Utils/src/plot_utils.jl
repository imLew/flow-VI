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

function plot_2D_results(initial_dist::Distribution, 
                         target_dist::Distribution, q)
    plt = plot()
    plot_2D_results!(plt, initial_dist, target_dist, q)
    return plt
end

function plot_2D_gaussians_results!(plt, data)
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    plot_2D_results!(plt, initial_dist, target_dist, 
                     data[:svgd_results][1][end]); 
end

function plot_2D_gaussians_results(data)
    plt = plot()
    plot_2D_gaussians_results!(plt, data)
    return plt
end

function plot_convergence(data; size=(375,375), legend=:bottomright, ylims=(-Inf,Inf), lw=3)
    sp1, sp2, sp3 = plot(), plot(), plot()
    layout = @layout [ i ; n b]
    plt = plot(sp1, sp2, sp3, layout=layout)
    plot_convergence!(plt, data, size=size, legend=legend, ylims=ylims, lw=lw)
    return plt
end

function plot_convergence!(plt::Plots.Plot, data; size=(375,375), 
                           legend=:bottomright, ylims=(-Inf,Inf), 
                           lw=3, int_lims=(-Inf,Inf))

    dist_plot = plot_2D_gaussians_results(data)
    
    norm_plot = plot(data[:svgd_results][1][1][:ϕ_norm],ylims=(0,Inf),
                     markeralpha=0, label="", title="", 
                     xticks=0:data[:n_iter]÷4:data[:n_iter], color=colors[1],
                     xlabel="iterations", ylabel="||φ||");

    layout = @layout [ i ; n b]
    final_plot = plot(int_plot, norm_plot, dist_plot, layout=layout, legend=:bottomright, size=size);
    for (sp, tp) in zip(plt.subplots, final_plot.subplots)
        merge_series!(sp, tp)
    end
end

function plot_integration(data; size=(375,375), legend=:bottomright, lw=3, 
                          ylims=(-Inf,Inf))
    plt = plot()
    plot_integration!(plt, data; size=size, legend=legend, 
                      lw=lw, ylims=ylims)
    return plt
end
export plot_integration!
function plot_integration!(plt::Plots.Plot, data; size=(375,375),
                           legend=:bottomright, lw=3, ylims=(-Inf,Inf))
    dKL_hist = data[:svgd_results][1][1]
    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)
    plot!(plt, xlabel="iterations", ylabel="log Z", legend=legend, lw=lw, ylims=ylims);
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_rkhs)[2]))
    plot!(plt, est_logZ, label="", color=colors[1]);
    hline!(plt, [true_logZ], labels="", color=colors[2], ls=:dash);
end

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
    append!(sp1.series_list, sp2.series_list)
    Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
    Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
end

# function plot_known_dists(initial_dist, target_dist, alg_params, 
#                       H₀, logZ, EV, dKL, q)
#     # caption="""n_particles=$n_particles; n_iter=$n_iter; 
#     #         norm_method=$norm_method; kernel_width=$kernel_width; 
#     #         step_size=$step_size"""
#     caption = ""
#     caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
#                       ticks=([]),fgborder=:white, subplot=1, framestyle=:none);
#     # title = """$(typeof(initial_dist)) $(Distributions.params(initial_dist)) 
#     #          target $(typeof(target_dist)) 
#     #          $(Distributions.params(target_dist))"""
#     title = ""
#     title_plot = plot(grid=false,annotation=(0.5,0.5,title),
#                       ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
#     int_plot, norm_plot = plot_integration(H₀, logZ, EV, dKL, 
#                                            alg_params[:step_size])

#     dim = size(q)[1]
#     if dim > 3 
#         layout = @layout [t{0.1h} ; d{0.3w} i ; c{0.1h}]
#         display(plot(title_plot, norm_plot, int_plot, 
#                       caption_plot, layout=layout, size=(1400,800), 
#                       legend=:topleft));
#     else
#         if dim == 1
#             dist_plot = plot_1D(initial_dist, target_dist, q)
#         elseif dim == 2
#             dist_plot = plot_2D(initial_dist, target_dist, q)
#         # elseif dim == 3
#         #     dist_plot = plot_3D(initial_dist, target_dist, q)
#         end
#     layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]
#     plot(title_plot, int_plot, norm_plot, dist_plot, 
#          caption_plot, layout=layout, size=(1400,800), 
#          legend=:topleft);
#     end
# end

