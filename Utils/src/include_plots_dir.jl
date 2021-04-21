using InteractiveUtils

export make_include_string

function trim_filename(file)
    file = split(file, "/")
    i = 1
    while file[i] != "plots"
        popfirst!(file)
    end
    join(file,"/")
end

function make_include_string(basedir)
    output_string = ""
    # for dirname in readdir(basedir, join=true)
        # dirname = "/home/lew/Documents/BCCN_Master/SVGD-stuff/Thesis/bayesian-inference-thesis/texfiles/plots/gauss/method_compare/" * dir

    plot_string = """
    \\begin{figure}[h]
        \\centering
        \\begin{tabular} """

    files = readdir(basedir, join=true)

    L = length(files)
    if L == 2 || L == 4
        plot_string *= "{cc}\n"
    elseif L == 3 || L === 6
        plot_string *= "{ccc}\n"
    end

    for (i, file) in enumerate(files)
        file = trim_filename(file)
        plot_string *= "\t\t\\includegraphics[width=.28\\textwidth]{$file} "
        if i == L
            plot_string *= "\n"
        elseif L/i == 2
            plot_string *= "\\\\ \n"
        else
            plot_string *= "& \n"
        end
    end

    plot_string *= """
        \\end{tabular}
        \\caption{TODO}
        \\label{fig:TODO}
    \\end{figure}

    """

    output_string *= plot_string
    # end

    clipboard(output_string)
    return output_string
end
