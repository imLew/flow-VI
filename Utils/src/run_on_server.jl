using BSON
using DrWatson

export cmdline_run

function cmdline_run(ALG_PARAMS, PROBLEM_PARAMS, DIRNAME, run_func)
    if length(ARGS) == 0 
        @info "Number of tasks defined by current params" ( dict_list_count(PROBLEM_PARAMS) * dict_list_count(ALG_PARAMS) )
        if haskey(ENV, "SGE_TASK_ID")
    # for running in a job array on the gridengine cluster;
    # assumes dictionaries with parameters have been created in _research/tmp
    # and that they are indexed in tmp_dict_names.bson
        dict_o_dicts = BSON.load(
                        projectdir("_research","tmp",
                            BSON.load(
                                  projectdir("tmp_dict_names.bson")
                            )[ENV["SGE_TASK_ID"]][1]
                        )
                       )
        @info "Sampling problem: $(dict_o_dicts[:problem_params])"
        @info "Alg parameters: $(dict_o_dicts[:alg_params])"
        @time run_func(problem_params=dict_o_dicts[:problem_params],
                      alg_params=dict_o_dicts[:alg_params],
                      DIRNAME)
        end
    elseif ARGS[1] == "make-dicts" 
    # make dictionaries in a tmp directory containing the parameters for
    # all the experiment we want to run
    # also saves a dictionary mapping numbers 1 through #dicts to the dictionary
    # names to index them
        dnames = Dict()
        for (i, alg_params) ∈ enumerate(dict_list(ALG_PARAMS))
            for (j, problem_params) ∈ enumerate(dict_list(PROBLEM_PARAMS))
                dname = tmpsave([@dict alg_params problem_params])
                dnames["$((i-1)*dict_list_count(PROBLEM_PARAMS) + j )"] = dname
            end
        end
        bson(projectdir("_research", "tmp_dict_names.bson"), dnames)
    elseif ARGS[1] == "run"
    # run the algorithm on the params specified in the second argument (bson)
        dict_o_dicts = BSON.load(ARGS[2])
        @info "Sampling problem: $(dict_o_dicts[:problem_params])"
        @info "Alg parameters: $(dict_o_dicts[:alg_params])"
        @time run_func(problem_params=dict_o_dicts[:problem_params],
                      alg_params=dict_o_dicts[:alg_params],
                      DIRNAME=DIRNAME)
    elseif ARGS[1] == "run-all"
        files = readdir(projectdir("_research", "tmp"), join=true)
        @info "Number of tmp files to run" length(files)
        Threads.@threads for (i, file) in enumerate(files)
            @info "experiment $i out of $(length(files))"
            try 
                run(`julia $PROGRAM_FILE run $file`)
            catch e
                println(e)
            end
        end
    elseif ARGS[1] == "make-and-run-all"
    # make the files containig the parameter dicts and start running them immediatly
        run(`julia $PROGRAM_FILE make-dicts`)
        run(`julia $PROGRAM_FILE run-all`)
    end
end

