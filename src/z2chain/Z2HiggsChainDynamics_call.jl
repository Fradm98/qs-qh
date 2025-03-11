include("Z2HiggsChainDynamics.jl")
using .Z2HiggsChainDynamics

using ArgParse
using Printf

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "L"
            help = "Length of the chain"
            arg_type = Int
            required = true
        "J"
            help = "J coupling constant"
            arg_type = Float64
            required = true
        "h"
            help = "h coupling constant"
            arg_type = Float64
            required = true
        "lamb"
            help = "lambda coupling constant"
            arg_type = Float64
            required = true
        "pp_left_pos"
            help = "Left position of the particle pair"
            arg_type = Int
            required = true
        "pp_length"
            help = "Length of the particle pair string"
            arg_type = Int
            required = true
        "final_time"
            help = "Final time for the evolution"
            arg_type = Float64
            required = true
        "dt"
            help = "Time step for the evolution"
            arg_type = Float64
            required = true
        "steps"
            help = "Number of steps to be plotted in the evolution"
            arg_type = Int
            required = true
        "--x_basis", "-x"
            help = "To use the x basis for the evolution"
            action = :store_true
        "--even", "-e"
            help = "Force even number of Trotter layers in the evolution"
            action = :store_true
        "--cutoff", "-c"
            help = "Bond dimension cutoff for the MPS"
            arg_type = Float64
            default = 1e-12
        "--folder", "-f"
            help = "Folder path where to save the results file"
            arg_type = String
            default = pwd()
        "--overwrite", "-o"
            help = "To overwrite the results file if it exists"
            action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    sites = Z2HiggsChainDynamics.chain_sites(args["L"])
    observables = Z2HiggsChainDynamics.all_local_pauli_z(sites)
    filename = @sprintf("tnjulia_z2pairquench_maxt_%f_dt_%f_L_%d_J_%.04f_h_%.04f_lamb_%.04f_pp_%d_xbasis_%s_steps_%d_cutoff_%.04f.npy",
                        args["final_time"], args["dt"], args["L"], args["J"], args["h"], args["lamb"],
                        args["pp_left_pos"], string(args["x_basis"]), args["steps"], args["cutoff"])
    simulation_filepath = joinpath(args["folder"], filename)
    Z2HiggsChainDynamics.particle_pair_quench_simulation(sites, observables, args["J"], args["h"], args["lamb"],
                                                        args["pp_left_pos"], args["pp_length"], args["final_time"],
                                                        args["dt"], args["steps"], args["x_basis"], args["even"],
                                                        args["cutoff"], simulation_filepath)
end

main()