module Z2HiggsChainDynamics

    export chain_sites, local_pauli_z, all_local_pauli_z, particle_pair_quench_simulation

    using ITensors, ITensorMPS
    using NPZ

    Sites = Vector{Index{Int64}}

    function nlayers(t::Real, error::Real, chain_length::Int, J::Real, h::Real, λ::Real)
        return Int(maximum([1, ceil(t .* (4/3*abs(J^2*λ) + 2/3*abs(J*h*λ) + 1/3*abs(h^2*λ) + abs(J*λ^2)/3 + 4*abs(h*λ^2)/6)*(chain_length-1)/error^(1/3))]))
    end

    function trotter_layer(sites::Sites, J::Real, h::Real, λ::Real, time::Real; x_basis::Bool=false)
        sq_rotations_exp = ITensor[]
        sqrotation_generator_str = x_basis ? "X" : "Z"
        for i in eachindex(sites)
            if i % 2 == 1
                this_generator = J*op(sites[i], sqrotation_generator_str)
            else
                this_generator = h*op(sites[i], sqrotation_generator_str)
            end
            this_exp = exp(-im * time / 2 * this_generator)
            push!(sq_rotations_exp, this_exp)
        end

        interaction_exp = ITensor[]
        sqinteraction_generator_str = x_basis ? "Z" : "X"
        interaction_site_inds = [i for i in 1:2:length(sites)-1 if i % 2 == 1]
        interaction_site_inds_odd = [i for (arr_ind, i) in pairs(interaction_site_inds) if arr_ind % 2 == 1]
        interaction_site_inds_even = [i for (arr_ind, i) in pairs(interaction_site_inds) if arr_ind % 2 == 0]
        for int_layer in [interaction_site_inds_odd, interaction_site_inds_even]
            this_layer_exp = ITensor[]
            for first_int_ind in int_layer
                this_interaction_generator = λ*op(sites[first_int_ind], sqinteraction_generator_str)*op(sites[first_int_ind+1], sqinteraction_generator_str)*op(sites[first_int_ind+2], sqinteraction_generator_str)
                this_exp = exp(-im * time * this_interaction_generator)
                push!(this_layer_exp, this_exp)
            end
            append!(interaction_exp, this_layer_exp)
        end

        trotter_layer = ITensor[]
        append!(trotter_layer, sq_rotations_exp)
        append!(trotter_layer, interaction_exp)
        append!(trotter_layer, sq_rotations_exp)
        return trotter_layer
    end

    function local_pauli_z(sites::Sites, ind::Int)
        return MPO(sites, [(i == ind) ? "Z" : "I" for i in 1:length(sites)])
    end

    function all_local_pauli_z(sites::Sites)
        return [local_pauli_z(sites, i) for i in 1:length(sites)]
    end

    function chain_sites(chain_length::Int)
        if chain_length < 2
            throw(ArgumentError("chain_length must be ≥ 2"))
        end
        siteinds("Qubit", 2*chain_length-1)
    end

    function particle_pair_quench_simulation(sites::Sites, observables::Vector{MPO}, J::Real, h::Real, λ::Real, particle_pair_left_position::Int, particle_pair_length::Int, final_time::Real, steps::Int, error::Real; x_basis::Bool=false, filepath::Union{String, Nothing}=nothing)
        if length(sites) < 2
            throw(ArgumentError("chain_length must be ≥ 2"))
        end

        if filepath !== nothing
            if isfile(filepath)
                site_gauge_observable_matrix = npzread(filepath)
                return site_gauge_observable_matrix
            end
        end

        # Initial state
        initial_state_string = ["Up" for i in 1:length(sites)]
        initial_state_string[2*(particle_pair_left_position)+1:2*particle_pair_left_position+1+2*particle_pair_length] .= "Dn"
        ψ0 = MPS(sites, initial_state_string)

        # Propagator
        t_per_step = final_time / (steps + 1)
        layers_per_step = nlayers(t_per_step, error, (length(sites) + 1) ÷ 2, J, h, λ)
        base_trotter_layer = trotter_layer(sites, J, h, λ, t_per_step / layers_per_step, x_basis=x_basis)
        step_propagator = ITensor[]
        for i in 1:layers_per_step
            append!(step_propagator, base_trotter_layer)
        end

        # Time evolution
        site_gauge_observable_matrix = zeros(Float64, (steps+1, length(observables)))
        ψ = ψ0
        for i in 0:steps
            this_time = (i-1) / steps * final_time
            print("\rt = $this_time")
            for (j, observable) in pairs(observables)
                site_gauge_observable_matrix[i+1, j] = real(inner(ψ', observable, ψ))
            end
            if i !== steps+1
                ψ = apply(step_propagator, ψ)
            end
        end

        print("t = $final_time")

        if filepath !== nothing
            npzwrite(filepath, site_gauge_observable_matrix)
        end

        return site_gauge_observable_matrix
    end
end