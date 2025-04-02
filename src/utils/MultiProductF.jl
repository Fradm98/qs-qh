module MultiProductF

    using Combinatorics
    using Convex, SCS
    using Statistics

    function mpfcoeffs(ks::Vector{Int}, pf_order::Int; print_mode::Bool=false)
        # if length(ks) != pf_order + 1
        #     throw(ArgumentError("Length of ks must be pf_order + 1"))
        # end

        a = Variable(length(ks))
        constraint_powers = zeros(length(ks))
        constraint_powers[2:end] .= pf_order .+ (2 .* collect(0:length(ks)-2))
        constraint_matrix = repeat(ks', length(constraint_powers), 1)
        constraint_matrix = constraint_matrix .^ reshape(-constraint_powers, length(constraint_powers), 1)
        constraint_values = zeros(length(ks))
        constraint_values[1] = 1
        problem = minimize(norm(a, 1), [constraint_matrix * a == constraint_values])
        # problem = minimize(sum(abs.(a)/(ks .^ 2*pf_order)), [constraint_matrix * a == constraint_values])
        solve!(problem, SCS.Optimizer, silent=!print_mode)
        return a.value, problem.optval
    end

    function search_mpf_coefficients(min_k::Int, max_k::Int, nks::Int, pf_order::Int)
        if min_k < 1
            throw(ArgumentError("min_k must be ≥ 1"))
        end
        if min_k > max_k
            throw(ArgumentError("min_k must be < max_k"))
        end
        
        k_candidates = reshape(vcat(collect(combinations(collect(min_k:max_k), nks))...), :, nks)
        coefficients = zeros(size(k_candidates)...)
        condition_numbers = zeros(size(k_candidates)[1])

        for i in 1:size(k_candidates)[1]
            coeffs, cond_number = mpfcoeffs(k_candidates[i, :], pf_order)
            coefficients[i, :] = coeffs
            condition_numbers[i] = cond_number
        end
    
        sorted_indices = sortperm(condition_numbers)
        coefficients = coefficients[sorted_indices, :]
        condition_numbers = condition_numbers[sorted_indices]
        k_candidates = k_candidates[sorted_indices, :]
    
        return k_candidates, coefficients, condition_numbers
    end
    
    function search_mpf_parameters_ccost(k_min::Int, k_max::Int, nks::Int, pf_order::Int, cost_func::Function; return_costs::Bool=false)
        if k_min > k_max
            throw(ArgumentError("k_max must be greater than ceil(t)"))
        end
    
        k_candidates, coefficients, condition_numbers = search_mpf_coefficients(k_min, k_max, nks, pf_order)
        cost_func_values = [cost_func(k_candidates[i, :], coefficients[i, :]) for i in 1:size(k_candidates)[1]]
        sorted_indices = sortperm(cost_func_values)
        k_candidates = k_candidates[sorted_indices, :]
        coefficients = coefficients[sorted_indices, :]
        condition_numbers = condition_numbers[sorted_indices]
        
        if return_costs
            return k_candidates, coefficients, condition_numbers, cost_func_values[sorted_indices]
        else
            return k_candidates, coefficients, condition_numbers
        end
    end

    function a_onenorm_cost(ks::Vector{Int}, coefficients::Vector{Float64})
        if length(ks) != length(coefficients)
            throw(ArgumentError("ks and coefficients must have the same length"))
        end
        return sum(abs.(coefficients))
    end

    function a_disp_onenorm_cost(ks::Vector{Int}, coefficients::Vector{Float64})
        if length(ks) != length(coefficients)
            throw(ArgumentError("ks and coefficients must have the same length"))
        end
        return sum(abs.(coefficients))*sqrt(sum((coefficients .- mean(coefficients)) .^ 2))
    end

    function k_a_onenorm_cost(ks::Vector{Int}, coefficients::Vector{Float64})
        if length(ks) != length(coefficients)
            throw(ArgumentError("ks and coefficients must have the same length"))
        end
        return sum(abs.(ks)) * sum(abs.(coefficients))
    end

    function k_a_disp_onenorm_cost(ks::Vector{Int}, coefficients::Vector{Float64})
        if length(ks) != length(coefficients)
            throw(ArgumentError("ks and coefficients must have the same length"))
        end
        return sum(abs.(ks)) * sum(abs.(coefficients))*sqrt(sum((coefficients .- mean(coefficients)) .^ 2))
    end

    function dt_limited_max_k(t::Real, dt::Real; min_k_max::Int=0)
        if t < 0
            throw(ArgumentError("t must be non-negative"))
        end
        if dt <= 0
            throw(ArgumentError("dt must be positive"))
        end
        suggested_k_max = ceil(Int, t / dt)
        if suggested_k_max < min_k_max
            return min_k_max
        else
            return suggested_k_max
        end
    end
end