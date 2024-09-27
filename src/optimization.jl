

#####
#Composite loss functions.
#####
function lossFunc(xyznew::Matrix{Float64}, lengths::Vector{Float64}, forces::Vector{Float64}, receiver::Receiver{TParams, TAnchorParams}, q::Vector{Float64}) where {TParams, TAnchorParams}
    loss = 0.0
    loss += qBounds(q, receiver.Params.UB, receiver.Params.LB)
    for obj in receiver.Params.Objectives
        loss += computeObjectiveLoss(obj, xyznew, lengths, forces)
    end
    return loss
end

function qBounds(q::Vector{Float64}, ub::Vector{Float64}, lb::Vector{Float64})
    return pBounds(q, ub, lb)
end


function computeObjectiveLoss(obj::Objective{TIndices, TValues}, xyznew, lengths, forces) where {TIndices<:AbstractVector{Int64}, TValues}
    if obj.ID == 1
        return obj.W * target(xyznew, obj.Values::Matrix{Float64}, obj.Indices)
    elseif obj.ID == 2
        return obj.W * lenVar(lengths, obj.Indices)
    elseif obj.ID == 3
        return obj.W * forceVar(forces, obj.Indices)
    elseif obj.ID == 4 #∑FL
        return obj.W * dot(lengths, forces)
    elseif obj.ID == 5 #"MinLength"
        return obj.W * minPenalty(lengths, obj.Values::Vector{Float64}, obj.Indices, 10.0)
    elseif obj.ID == 6 #"MaxLength"
        return obj.W * maxPenalty(lengths, obj.Values::Vector{Float64}, obj.Indices, 10.0)
    elseif obj.ID == 7 #"MinForce"
        return obj.W * minPenalty(forces, obj.Values::Vector{Float64}, obj.Indices, 10.0)
    elseif obj.ID == 8 #"MaxForce"
        return obj.W * maxPenalty(forces, obj.Values::Vector{Float64}, obj.Indices, 10.0)
    elseif obj.ID == 9 #"TargetLen"
        return obj.W * lenTarget(lengths, obj.Values::Vector{Float64}, obj.Indices)
    else
        return 0.0
    end
end