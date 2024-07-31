

#####
#Composite loss functions.
#####
function lossFunc(xyznew::Matrix{Float64}, lengths::Vector{Float64}, forces::Vector{Float64}, receiver::Receiver, q::Vector{Float64})
    loss = 0.0
    #Enforce parameter bounds on q to prevent negative values
    #loss += pBounds(q, receiver.Params.LB, receiver.Params.UB)

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

#####
#Composite loss functions.
#####
function lossFunc(xyznew::Matrix{Float64}, lengths::Vector{Float64}, forces::SparseVector{Float64, Int64}, receiver::Receiver, q::SparseVector{Float64, Int64})
    loss = 0.0


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


# Define the L-BFGS optimization function with custom line search
function lbfgs_custom(f, x0; m=10, max_iter=100, tol=1e-6, min_val=1e-8)
    n = length(x0)
    x = max.(x0, min_val)
    s = zeros(n, m)
    y = zeros(n, m)
    rho = zeros(m)
    alpha = zeros(m)
    fval = f(x)
    g = x -> Zygote.gradient(f, x)[1]
    gval = g(x)
    q = copy(gval)
    k = 0
    
    while norm(gval) > tol && k < max_iter
        # Two-loop recursion to compute search direction
        if k > 0
            for i in (k-1):-1:max(k-m, 1)
                alpha[i % m + 1] = rho[i % m + 1] * dot(s[:, i % m + 1], q)
                q -= alpha[i % m + 1] * y[:, i % m + 1]
            end
            gamma = dot(s[:, (k-1) % m + 1], y[:, (k-1) % m + 1]) / dot(y[:, (k-1) % m + 1], y[:, (k-1) % m + 1])
            r = gamma * q
            for i in max(k-m, 1):k-1
                beta = rho[i % m + 1] * dot(y[:, i % m + 1], r)
                r += s[:, i % m + 1] * (alpha[i % m + 1] - beta)
            end
        else
            r = -gval
        end

        # Custom line search to ensure non-negative parameter values
        d = -r
        best_alpha, fval_new, x_new = gd_line_search(f, g, x, d, 10.0; max_iter=receiver.Params.MaxIter, min_val=min_val)

        s[:, k % m + 1] = x_new - x
        x = x_new
        gval_new = g(x)
        y[:, k % m + 1] = gval_new - gval
        rho[k % m + 1] = 1.0 / dot(y[:, k % m + 1], s[:, k % m + 1])
        gval = gval_new

        #If the change in loss value is small, just return
        if k > 0 && abs(fval - fval_new) < tol
            return x, fval, norm(gval) <= tol, k
        end

        x .= x_new # normalize and update parameters
        fval = fval_new #update fval

        k += 1 #increment counter

    end
    
    return x, fval, norm(gval) <= tol, k
end