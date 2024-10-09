import HTTP.WebSockets
using Optim
using JSON

"""
Perform Force Density Method (FDM) Optimization.

# Arguments
- `receiver::Receiver`: The receiver object containing network and optimization parameters.
- `ws::WebSocket`: The WebSocket connection for sending updates.

# Description
This function performs optimization on the network's force densities using the Force Density Method (FDM).
It handles both cases where objectives are provided and where they're absent, sending progress updates 
via WebSockets throughout the optimization process.
"""
function FDMoptim!(receiver::Receiver, ws::WebSocket)
    # Initialize sparse point index
    sp_init = collect(Int64, 1:receiver.ne)

    # Objective function absent: solve explicitly and send final state
    if isnothing(receiver.Params) || isnothing(receiver.Params.Objectives) || isempty(receiver.Params.Objectives)
        println("SOLVING")

        xyznew = solve_explicit(receiver.Q, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf, sp_init)

        # Preallocate the full XYZ matrix
        xyz = zeros(Float64, receiver.nn, 3)
        @inbounds begin
            xyz[receiver.N, :] .= xyznew
            xyz[receiver.F, :] .= receiver.XYZf
        end

        # Prepare the output message
        msgout = Dict(
            "Finished"    => true,
            "Iter"        => 1,
            "Loss"        => 0.0,
            "Q"           => receiver.Q,
            "X"           => xyz[:, 1],
            "Y"           => xyz[:, 2],
            "Z"           => xyz[:, 3],
            "Losstrace"   => [0.0]
        )

        # Send the message via WebSocket
        HTTP.WebSockets.send(ws, JSON.json(msgout))
        return
    end

    println("OPTIMIZING")

    # Initialize trace variables
    Q_trace = Vector{Float64}()
    NodeTrace = Vector{Matrix{Float64}}()
    iters = Vector{Vector{Float64}}(undef, receiver.Params.MaxIter)
    losses = Vector{Float64}(undef, receiver.Params.MaxIter)
    counter = 0  # To track iterations

    """
    Objective function when Anchor Parameters are present.
    """
    function obj_xyz(p::Vector{Float64})
        q = p[1:receiver.ne]
        init_params = receiver.AnchorParams.Init
        # Concatenate new and old XYZf directly since ordering is ensured
        xyzf = vcat(reshape(p[receiver.ne+1:end], (:, 3)), receiver.XYZf)

        xyznew = solve_explicit(q, receiver.Cn, receiver.Cf, receiver.Pn, xyzf, sp_init)
        xyzfull = vcat(xyznew, xyzf)

        # Calculate lengths and forces
        lengths = norm.(eachrow(receiver.C * xyzfull))
        forces = q .* lengths

        # Deep copy for trace without affecting AD
        if !isderiving()
            ignore_derivatives() do
                Q_trace .= copy(q)  # Reuse the existing vector for trace
                if receiver.Params.NodeTrace
                    push!(NodeTrace, copy(xyzfull))
                end
            end
        end

        # Compute loss
        loss = lossFunc(xyzfull, lengths, forces, receiver, q)
        return loss
    end

    """
    Objective function when Anchor Parameters are absent.
    """
    function obj(q::Vector{Float64})
        xyznew = solve_explicit(q, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf, sp_init)
        xyzfull = vcat(xyznew, receiver.XYZf)

        lengths = norm.(eachrow(receiver.C * xyzfull))
        forces = q .* lengths

        # Compute loss
        loss = lossFunc(xyznew, lengths, forces, receiver, q)

        if !isderiving()
            ignore_derivatives() do
                Q_trace .= copy(q)
                if receiver.Params.NodeTrace
                    push!(NodeTrace, copy(xyzfull))
                end

                counter += 1

                if receiver.Params.Show && (counter % receiver.Params.Freq == 0)
                    iters[counter] = copy(Q_trace)
                    losses[counter] = loss

                    # Prepare intermediate message
                    msgout = Dict(
                        "Finished"     => false,
                        "Iter"         => counter,
                        "Loss"         => loss,
                        "Q"            => copy(Q_trace),
                        "X"            => receiver.XYZf[:, 1],
                        "Y"            => receiver.XYZf[:, 2],
                        "Z"            => receiver.XYZf[:, 3],
                        "Losstrace"    => losses[1:counter]
                    )

                    if receiver.Params.NodeTrace
                        msgout["NodeTrace"] = NodeTrace
                    end

                    # Send the message via WebSocket
                    HTTP.WebSockets.send(ws, JSON.json(msgout))
                end
            end
        end

        return loss
    end

    """
    Callback function for optimization.
    """
    function cb(loss::Float64)
        global cancel
        if cancel
            cancel = false
            return true
        end

        if receiver.Params.Show
            iters[counter + 1] = copy(Q_trace)
            losses[counter + 1] = loss

            # Prepare intermediate message
            msgout = Dict(
                "Finished"   => false,
                "Iter"       => counter + 1,
                "Loss"       => loss,
                "Q"          => copy(Q_trace),
                "X"          => receiver.XYZf[:, 1],
                "Y"          => receiver.XYZf[:, 2],
                "Z"          => receiver.XYZf[:, 3],
                "Losstrace"  => losses[1:counter + 1]
            )

            if receiver.Params.NodeTrace
                msgout["NodeTrace"] = NodeTrace
            end

            # Send the message via WebSocket
            HTTP.WebSockets.send(ws, JSON.json(msgout))
            return false
        else
            return false
        end
    end

    """
    Gradient function using Zygote.
    """
    function g!(G::Vector{Float64}, θ::Vector{Float64})
        grad = gradient(θ) do q
            if isnothing(receiver.AnchorParams)
                obj(q)
            else
                obj_xyz(q)
            end
        end
        G .= grad[1]
    end

    """
    Placeholder for drape gradient function. To be implemented as needed.
    """
    function drape!(G::Vector{Float64}, θ::Vector{Float64})
        G .= 0.0  # Assuming no gradient contribution
    end

    # Set up objective and parameters based on Anchor Parameters presence
    if isnothing(receiver.AnchorParams)
        optimization_obj = obj
        parameters = copy(receiver.Q)  # Ensure a mutable copy
    else
        optimization_obj = obj_xyz
        parameters = vcat(receiver.Q, receiver.AnchorParams.Init)
    end

    # Perform optimization using Optim.jl
    res = Optim.optimize(
        optimization_obj,
        g!,
        parameters,
        LBFGS(),
        Optim.Options(
            iterations = receiver.Params.MaxIter,
            f_tol      = receiver.Params.RelTol
        )
    )

    minimized_params = Optim.minimizer(res)

    println("------------------------------------")
    println("Optimizer: ", Optim.summary(res))
    println("Iterations: ", Optim.iterations(res))
    println("Function calls: ", Optim.f_calls(res))
    println("SOLUTION FOUND")

    # Parse the solution
    if isnothing(receiver.AnchorParams)
        xyz_final = solve_explicit(
            minimized_params,
            receiver.Cn,
            receiver.Cf,
            receiver.Pn,
            receiver.XYZf,
            sp_init
        )
        xyz_final = vcat(xyz_final, receiver.XYZf)
    else
        newXYZf = reshape(minimized_params[1:receiver.ne], (:, 3))
        # Concatenate new and old XYZf directly
        xyzf_final = vcat(newXYZf, receiver.XYZf)
        xyz_final = solve_explicit(
            minimized_params[1:receiver.ne],
            receiver.Cn,
            receiver.Cf,
            receiver.Pn,
            xyzf_final,
            sp_init
        )
        xyz_final = vcat(xyz_final, xyzf_final)
    end

    # Prepare final output message
    msgout = Dict(
        "Finished"   => true,
        "Iter"       => counter,
        "Loss"       => Optim.minimum(res),
        "Q"          => minimized_params[1:receiver.ne],
        "X"          => xyz_final[:, 1],
        "Y"          => xyz_final[:, 2],
        "Z"          => xyz_final[:, 3],
        "Losstrace"  => losses[1:counter]
    )

    if receiver.Params.NodeTrace
        msgout["NodeTrace"] = NodeTrace
    end

    # Send the final message via WebSocket
    HTTP.WebSockets.send(ws, JSON.json(msgout))
end