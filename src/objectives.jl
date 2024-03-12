"""
Minimze distances between selected target nodes and their corresponding nodes in the form found network.
"""
function target(xyz, target, indices)
    sum((xyz[indices,:] - target).^2)
end


"""
Penalize values to be between lb and ub with a smooth approximation of ReLU. 
Scaling parameters kL and kU can be introduced to make the inflection points of the upper and lower bounds more precise.
"""

function pBounds(p::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})
    sum(softplus(-10*p - lb)) + sum(softplus(p - ub))
end

function pBounds(p::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, kL::Float64, kU::Float64)
    sum(softplus(-10*(p - lb) , kL)) + sum(softplus((p - ub), kU))
end


"""
Compute the maximum and minimum lengths of the edges in the network using softmax and softmin dotted with the edge lengths.
"""
function lenVar(x::Vector{Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}})
    x = x[indices]
    max = x' * softmax(x)
    min = x' * softmin(x)
    max - min
end

"""
Reduce the difference between the maximum and minimum forces in the network.
From Schek theorem 2. 
"""
function forceVar(x::Vector{Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}})
    x = x[indices]
    sum(x)
    #-reduce(-, extrema(x))
end

"""
Minimize the difference between the form found lengths of the edges and the target lengths.
"""

function lenTarget(lengths::Vector{Float64}, values::Vector{Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}})
    sum((lengths[indices] - values).^2)
end

"""
Penalizes values in vector that are below a threshold 
"""
function minPenalty(x::Vector{Float64}, values::Union{Vector{Float64},Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}})
    x = x[indices]
    sum(softplus.(-x .- values))
end

function minPenalty(x::Vector{Float64}, values::Union{Vector{Float64},Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}}, k::Union{Float64,Int64})
    x = x[indices]
    sum(softplus.((x - values), k))
end

"""
Penalizes values in vector that are above a threshold 
"""

function maxPenalty(x::Vector{Float64}, values::Union{Vector{Float64},Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}})
    minPenalty(-x, values, indices)
end
function maxPenalty(x::Vector{Float64}, values::Union{Vector{Float64},Float64}, indices::Union{Vector{Int64}, UnitRange{Int64}}, k::Union{Float64,Int64})
    minPenalty(-x, values, indices, k)
end


"""
Cross entropy loss function.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64})
    -sum(t .* log1p.(p))
end

"""
Weighted cross entropy loss, best used with a plain mask vector of weights.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64}, w::Union{SparseVector{Float64, Int64}, Vector{Float64}})
    -vec(t .* log1p.(p))' * w
end

"""
Softplus is a smooth approximation of the ReLU function.
A sharpness parameter k may be included to make the inflection point more precise.
"""
function softplus(x)
    log1p.(1 .+ exp.(x))
end

function softplus(x, k)
    log1p.(1 .+ exp.(k*x)) ./ k
end

"""
The derivative of the softplus function is the logistic function.
A scaling parameter k can be introduced to make the inflection point more precise.
This is a smooth approximation of the heaviside step function. 
https://en.wikipedia.org/wiki/Logistic_function
"""
function logisticFunc(x)
    1 / (1 + exp.(-x))
end

function logisticFunc(x, k::Union{Float64,Int64})
    1 / (1 + exp.(-k*x))
end

"""
LogSumExp is the multivariable generalization of the logistic function.
"""
function logSumExp(x)
    log1p(sum(exp.(x)))
end

"""
Softmax is a generalized version of the logistic function.
It returns a vector of probabilities that sum to 1.
"""
function softmax(x)
    exp.(x) / sum(exp.(x))
end

function softmin(x)
    softmax(-x)
end


