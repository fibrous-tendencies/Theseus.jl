

#####
#Composite loss functions.
#####
function lossFunc(xyznew::Matrix{Float64}, lengths::Vector{Float64}, forces::Vector{Float64}, receiver::Receiver, q::Vector{Float64})
    loss = 0.0
    #Enforce parameter bounds on q to prevent negative values
    loss += pBounds(q, receiver.Params.LB, receiver.Params.UB)

    #evaluate objective and return composite loss
    for obj in receiver.Params.Objectives
        if obj.ID == "None"
            loss += 0.0
        elseif obj.ID == "Target"
            loss += obj.W * target(xyznew, obj.Values, obj.Indices)
        elseif obj.ID == "LengthVar"
            loss += obj.W * lenVar(lengths, obj.Indices)
        elseif obj.ID == "ForceVar"
            loss += obj.W * forceVar(lengths, obj.Indices)
        elseif obj.ID == "Performance" #∑FL
            loss += obj.W * dot(lengths, forces)
        elseif obj.ID == "MinLength"
            loss += obj.W * minPenalty(lengths, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == "MaxLength"
            loss += obj.W * maxPenalty(lengths, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == "MinForce"
            loss += obj.W * minPenalty(forces, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == "MaxForce"
            loss += obj.W * maxPenalty(forces, obj.Values, obj.Indices, 10.0)
        elseif obj.ID == "TargetLen"
            loss += obj.W * lenTarget(lengths, obj.Values, obj.Indices)
        end
    end
    
    return loss
end
