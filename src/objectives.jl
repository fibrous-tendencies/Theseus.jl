"""
Minimize distances between selected target nodes and their corresponding nodes in the form found network.
"""
function target(xyz::Matrix{Float64}, target::Matrix{Float64}, indices::Vector{Int64})::Float64
    local sum_sq = 0.0
    @inbounds for i in eachindex(indices)
        local diff = xyz[indices[i], :] .- target[i, :]
        @inbounds for j in eachindex(diff)
            sum_sq += diff[j]^2
        end
    end
    return sum_sq
end


"""
Penalize values to be between lb and ub with a smooth approximation of ReLU.
Uses a default scaling parameter for lower bounds.
"""
function pBounds(p::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, kL::Float64 = 10.0, kU::Float64 = 10.0)::Float64
    local penalty = 0.0
    @inbounds for i in eachindex(p)
        penalty += softplus(-kL * (p[i] - lb[i])) + softplus(kU * (p[i] - ub[i]))
    end
    return penalty
end


"""
Compute the variance in edge lengths by calculating the difference between the maximum and minimum lengths.

Returns:
    Float64 representing the range of lengths.
"""
function lenVar(x::Vector{Float64}, indices::Vector{Int64})::Float64
    local min_val = Inf
    local max_val = -Inf
    @inbounds for i in eachindex(indices)
        local val = x[indices[i]]
        if val < min_val
            min_val = val
        end
        if val > max_val
            max_val = val
        end
    end
    return max_val - min_val
end

"""
Compute the variance in forces by calculating the difference between the maximum and minimum forces.

Returns:
    Float64 representing the range of forces.
"""
function forceVar(x::Vector{Float64}, indices::Vector{Int64})::Float64
    local min_val = Inf
    local max_val = -Inf
    @inbounds for i in eachindex(indices)
        local val = x[indices[i]]
        if val < min_val
            min_val = val
        end
        if val > max_val
            max_val = val
        end
    end
    return max_val - min_val
end

"""
Minimize the squared difference between form-found lengths of edges and target lengths.

Returns:
    Float64 representing the sum of squared differences.
"""
function lenTarget(lengths::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64})::Float64
    local sum_sq = 0.0
    @inbounds for i in eachindex(indices)
        local diff = lengths[indices[i]] - values[i]
        sum_sq += diff * diff
    end
    return sum_sq
end

"""
Penalizes values in the vector that are below a specified threshold using a smooth approximation.

Arguments:
    p::Vector{Float64} - The parameter vector.
    values::Vector{Float64} - The threshold values.
    indices::Vector{Int64} - The indices to apply the penalty.
    k::Float64 - The scaling factor for the softplus function.

Returns:
    Float64 representing the total penalty.
"""
function minPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64 = 1.0)::Float64
    local penalty = 0.0
    @inbounds for i in eachindex(indices)
        local xi = x[indices[i]]
        local vi = values[i]
        penalty += softplus(k * (xi - vi))
    end
    return penalty
end

"""
Penalizes values in the vector that are above a specified threshold using a smooth approximation.

Arguments:
    p::Vector{Float64} - The parameter vector.
    values::Vector{Float64} - The threshold values.
    indices::Vector{Int64} - The indices to apply the penalty.
    k::Float64 - The scaling factor for the softplus function.

Returns:
    Float64 representing the total penalty.
"""
function maxPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64 = 1.0)::Float64
    local penalty = 0.0
    @inbounds for i in eachindex(indices)
        local xi = x[indices[i]]
        local vi = values[i]
        penalty += softplus(k * (xi - vi))
    end
    return penalty
end


"""
Compute the cross-entropy loss.

Arguments:
    t::Vector{Float64} - Target vector.
    p::Vector{Float64} - Prediction vector.

Returns:
    Float64 representing the cross-entropy loss.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64})::Float64
    local loss = 0.0
    @inbounds for i in eachindex(t)
        loss -= t[i] * log1p(p[i])
    end
    return loss
end

"""
Compute the weighted cross-entropy loss.

Arguments:
    t::Vector{Float64} - Target vector.
    p::Vector{Float64} - Prediction vector.
    w::Vector{Float64} - Weight vector.

Returns:
    Float64 representing the weighted cross-entropy loss.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64}, w::Vector{Float64})::Float64
    local loss = 0.0
    @inbounds for i in eachindex(t)
        loss -= t[i] * log1p(p[i]) * w[i]
    end
    return loss
end

"""
Compute the Softplus function, a smooth approximation of ReLU.

Arguments:
    x::Float64 - Input value.
    k::Float64 - Optional sharpness parameter (default is 1.0).

Returns:
    Float64 representing the softplus of the input.
"""
function softplus(x::Float64, k::Float64 = 1.0)::Float64
    return log1p(exp(k * x)) / k
end

"""
Compute the Softplus function element-wise for a vector.

Arguments:
    x::Vector{Float64} - Input vector.
    k::Float64 - Optional sharpness parameter (default is 1.0).

Returns:
    Float64 representing the sum of softplus applied to each element.
"""
function softplus_sum(x::Vector{Float64}, k::Float64 = 1.0)::Float64
    local sum_val = 0.0
    @inbounds for xi in x
        sum_val += log1p(exp(k * xi)) / k
    end
    return sum_val
end

"""
Compute the Logistic function, a smooth approximation of the Heaviside step function.

Arguments:
    x::Float64 - Input value.
    k::Float64 - Optional scaling parameter (default is 1.0).

Returns:
    Float64 representing the logistic function of the input.
"""
function logisticFunc(x::Float64, k::Float64 = 1.0)::Float64
    return 1.0 / (1.0 + exp(-k * x))
end

"""
Compute the Logistic function element-wise for a vector.

Arguments:
    x::Vector{Float64} - Input vector.
    k::Float64 - Optional scaling parameter (default is 1.0).

Returns:
    Vector{Float64} representing the logistic function applied to each element.
"""
function logisticFunc_vec(x::Vector{Float64}, k::Float64 = 1.0)::Vector{Float64}
    local y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = 1.0 / (1.0 + exp(-k * x[i]))
    end
    return y
end

"""
Compute the LogSumExp of a vector in a numerically stable way.

Arguments:
    x::Vector{Float64} - Input vector.

Returns:
    Float64 representing the log-sum-exp of the input.
"""
function logSumExp(x::Vector{Float64})::Float64
    local m = maximum(x)
    local sum_exp = 0.0
    @inbounds for xi in x
        sum_exp += exp(xi - m)
    end
    return m + log1p(sum_exp - 1.0)  # log1p(sum_exp - 1) = log(sum_exp)
end

"""
Compute the Softmax of a vector in a numerically stable way.

Arguments:
    x::Vector{Float64} - Input vector.

Returns:
    Vector{Float64} representing the softmax probabilities.
"""
function softmax(x::Vector{Float64})::Vector{Float64}
    local m = maximum(x)
    local sum_exp = 0.0
    local y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = exp(x[i] - m)
        sum_exp += y[i]
    end
    @inbounds for i in eachindex(y)
        y[i] /= sum_exp
    end
    return y
end

function softmin(x::Vector{Float64})::Vector{Float64}
    softmax(-x)
end


