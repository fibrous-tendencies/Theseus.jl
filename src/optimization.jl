

#####
#Composite loss functions.
#####
function lossFunc(xyznew::Matrix{Float64}, lengths::Vector{Float64}, forces::Vector{Float64}, receiver::Receiver, q::Vector{Float64})
    loss = 0.0
    #Enforce parameter bounds on q to prevent negative values
    loss += pBounds(q, receiver.Params.LB, receiver.Params.UB)

    #evaluate objective and return composite loss
    for obj in receiver.Params.Objectives
        if obj.ID == -1
            loss += 0.0
        elseif obj.ID == 1 #Target
            loss += obj.W * target(xyznew, obj.Values, obj.Indices)
        elseif obj.ID == 2 #Length Variation
            loss += obj.W * lenVar(lengths, obj.Indices)
        elseif obj.ID == 3 #Force Variation
            loss += obj.W * forceVar(lengths, obj.Indices)
        elseif obj.ID == 4 #∑FL
            loss += obj.W * dot(lengths, forces)
        elseif obj.ID == 5 #"MinLength"
            loss += obj.W * minPenalty(lengths, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == 6 #"MaxLength"
            loss += obj.W * maxPenalty(lengths, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == 7 #"MinForce"
            loss += obj.W * minPenalty(forces, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == 8 #"MaxForce"
            loss += obj.W * maxPenalty(forces, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == 9 #"TargetLen"
            loss += obj.W * lenTarget(lengths, obj.Values, obj.Indices)
        end
    end
    
    return loss
end
