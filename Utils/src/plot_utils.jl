using Plots
using Distributions
using ColorSchemes

const colors = ColorSchemes.seaborn_colorblind

# export plot_known_dists
export plot_2D
export plot_1D

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

function plot_2D(initial_dist::Distribution, target_dist::Distribution, q)
    dist_plot = scatter(q[1,:], q[2,:], legend=false, label="", msw=0.0, alpha=0.5, color=colors[1]);
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
    contour!(dist_plot, x, y, (x,y)->pdf(target_dist, [x, y]), color=colors[2], 
             label="", levels=5, msw=0.0, alpha=0.6)
    contour!(dist_plot, x, y, (x,y)->pdf(initial_dist, [x, y]), color=colors[1], 
             label="", levels=5, msw=0.0, alpha=0.6)
    return dist_plot
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

