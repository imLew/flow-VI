export plot_known_dists
export plot_2D
export plot_1D

function plot_known_dists(initial_dist, target_dist, alg_params, 
                      H₀, logZ, EV, dKL, q)
    # caption="""n_particles=$n_particles; n_iter=$n_iter; 
    #         norm_method=$norm_method; kernel_width=$kernel_width; 
    #         step_size=$step_size"""
    caption = ""
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white, subplot=1, framestyle=:none);
    # title = """$(typeof(initial_dist)) $(Distributions.params(initial_dist)) 
    #          target $(typeof(target_dist)) 
    #          $(Distributions.params(target_dist))"""
    title = ""
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    int_plot, norm_plot = plot_integration(H₀, logZ, EV, dKL, 
                                           alg_params[:step_size])

    dim = size(q)[1]
    if dim > 3 
        layout = @layout [t{0.1h} ; d{0.3w} i ; c{0.1h}]
        display(plot(title_plot, norm_plot, int_plot, 
                      caption_plot, layout=layout, size=(1400,800), 
                      legend=:topleft));
    else
        if dim == 1
            dist_plot = plot_1D(initial_dist, target_dist, q)
        elseif dim == 2
            dist_plot = plot_2D(initial_dist, target_dist, q)
        # elseif dim == 3
        #     dist_plot = plot_3D(initial_dist, target_dist, q)
        end
    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]
    plot(title_plot, int_plot, norm_plot, dist_plot, 
         caption_plot, layout=layout, size=(1400,800), 
         legend=:topleft);
    end
end

function plot_2D(initial_dist, target_dist, q)
    # TODO add 'show title' parameter?
    dist_plot = scatter(q[1,:], q[2,:], 
                        labels="q");
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.5*abs(min_q):0.05:max_q+0.5*abs(max_q)
    contour!(dist_plot, t, t, (x,y)->pdf(target_dist, [x, y]), color=:black, 
             label="p", levels=5)
    contour!(dist_plot, t, t, (x,y)->pdf(initial_dist, [x, y]), color=:black, 
             label="q_0", levels=5)
    return dist_plot
end

function plot_1D(initial_dist, target_dist, q)
    n_bins = length(q) ÷ 5
    dist_plot = histogram(reshape(q, length(q)), 
                          fillalpha=0.3, labels="q" ,bins=20,
                          normalize=true);
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.2*abs(min_q):0.05:max_q+0.2*abs(max_q)
    plot!(x->pdf(target_dist, x), t, labels="p")
    plot!(x->pdf(initial_dist, x), t, labels="q₀")
    return dist_plot
end

