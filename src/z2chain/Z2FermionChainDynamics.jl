module Z2FermionDynamics
    using ITensors, ITensorMPS
    using NPZ

    Sites = Vector{Index{Int64}}

    function local_pauli_z(sites::Sites, ind::Int)
        return MPO(sites, [(i == ind) ? "Z" : "I" for i in 1:length(sites)])
    end

    function all_local_pauli_z(sites::Sites)
        return [local_pauli_z(sites, i) for i in 1:length(sites)]
    end

    function trotter_layer(sites::Sites, J::Real, h::Real, λ::Real, time::Real; x_basis::Bool=false)
        
    end
end