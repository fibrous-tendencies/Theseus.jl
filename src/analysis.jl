import HTTP.WebSockets

### optimiztaion
function FDMoptim!(receiver, ws)

        sp_init = collect(Int64, range(1, length = receiver.ne))

        # objective function
        if isnothing(receiver.Params) || isnothing(receiver.Params.Objectives) || isempty(receiver.Params.Objectives)

            println("SOLVING")

            xyznew = solve_explicit(receiver.Q, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf, sp_init)

            xyz = zeros(receiver.nn, 3)
            xyz[receiver.N, :] = xyznew
            xyz[receiver.F, :] = receiver.XYZf            

            msgout = Dict("Finished" => true,
                    "Iter" => 1, 
                    "Loss" => 0.,
                    "Q" => receiver.Q, 
                    "X" => xyz[:,1], 
                    "Y" => xyz[:,2], 
                    "Z" => xyz[:,3],
                    "Losstrace" => [0.])

            HTTP.WebSockets.send(ws, json(msgout))
            
        else
            try
                
            
            println("OPTIMIZING")

            #trace
            Q = []
            NodeTrace = []
            
            i = 0
            iters = Vector{Vector{Float64}}()
            losses = Vector{Float64}()

            """
            Objective function, returns a scalar loss value wrt the parameters.
            """
            function obj_xyz(p)
                q = p[1:receiver.ne]

                newXYZf = reshape(p[receiver.ne+1:end], (:, 3))
                oldXYZf = receiver.XYZf[receiver.AnchorParams.FAI, :]

                xyzf = combineSorted(newXYZf, oldXYZf, receiver.AnchorParams.VAI, receiver.AnchorParams.FAI)

                xyznew = solve_explicit(q, receiver.Cn, receiver.Cf, receiver.Pn, xyzf, sp_init)

                xyzfull = vcat(xyznew, xyzf)
                
                lengths = norm.(eachrow(receiver.C * xyzfull))
                forces = q .* lengths

                if !isderiving()
                    ignore_derivatives() do
                        Q = deepcopy(q)
                        if receiver.Params.NodeTrace == true
                            push!(NodeTrace, deepcopy(xyzfull))
                        end
                        
                    end
                end

                loss = lossFunc(xyzfull, lengths, forces, receiver, q)

                return loss
            end          

            function obj(q)           

                xyznew = solve_explicit(q, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf, sp_init)
                
                xyzfull = vcat(xyznew, receiver.XYZf)  
                
                lengths = norm.(eachrow(receiver.C * xyzfull))
                forces = q .* lengths        

                loss = lossFunc(xyznew, lengths, forces, receiver, q)

                if !isderiving()
                    ignore_derivatives() do
                        Q = deepcopy(q)
                        if receiver.Params.Show && i % receiver.Params.Freq == 0
                            
                            # Check if loss is not null before updating losses
                            if loss !== nothing
                                push!(losses, loss)
                            end
                            # Send message with Losstrace
                            msgout = Dict("Finished" => false,
                                "Iter" => i, 
                                "Loss" => loss,
                                "Q" => Q, 
                                "X" => xyzfull[:,1], 
                                "Y" => xyzfull[:,2], 
                                "Z" => xyzfull[:,3],
                                "Losstrace" => losses)
                            HTTP.WebSockets.send(ws, json(msgout))
                        end
                    end
                end
                
                return loss
            end

            

            #callback function
            function cb(loss)

                 if cancel == true
                    global cancel = false
                    return true     
                
                if receiver.Params.Show 
                    # Check if loss.value is not null before updating losses
                    if loss.value !== nothing
                        push!(losses, loss.value)
                    end
                    # Send message with Losstrace
                    msgout = Dict("Finished" => false,
                        "Iter" => i, 
                        "Loss" => loss.value,
                        "Q" => Q, 
                        "X" => last(NodeTrace)[:,1], 
                        "Y" => last(NodeTrace)[:,2], 
                        "Z" => last(NodeTrace)[:,3],
                        "Losstrace" => losses)
                        
                    HTTP.WebSockets.send(ws, json(msgout))
                    return false
                    end
                else      
                    return false
                end

            end

            """
            Gradient function, returns a vector of gradients wrt the parameters.
            """

            function g!(G, θ)
                grad = gradient(θ) do q
                   obj(q)
                end
                G .= grad[1]
            end
            
            #todo add explicit gradient for distance conditions from Schek
            #to use when draping only
            function drape!(G, θ)
                grad = nothing
                G .= grad[1]
            end

            """
            Optimization
            """
            if isnothing(receiver.AnchorParams)
                obj = obj
                parameters = receiver.Q
            else
                obj = obj_xyz
                parameters = vcat(receiver.Q, receiver.AnchorParams.Init)
            end
            res = Optim.optimize( 
                obj, 
                g!,
                parameters,
                LBFGS(),                
                Optim.Options(
                    iterations = receiver.Params.MaxIter,
                    f_tol = receiver.Params.RelTol,
                    ))            

            min = Optim.minimizer(res)
    
            println("------------------------------------")
            println("Optimizer: ", summary(res))
            println("Iterations: ", Optim.iterations(res))
            println("Function calls: ", Optim.f_calls(res))


            println("SOLUTION FOUND")
            # PARSING SOLUTION
            if isnothing(receiver.AnchorParams)
                xyz_final = solve_explicit(min, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf, sp_init)
                xyz_final = vcat(xyz_final, receiver.XYZf)
            else
                newXYZf = reshape(min[receiver.ne+1:end], (:, 3))
                oldXYZf = receiver.XYZf[receiver.AnchorParams.FAI, :]
                XYZf_final = combineSorted(newXYZf, oldXYZf, receiver.AnchorParams.VAI, receiver.AnchorParams.FAI)
                xyz_final = solve_explicit(min[1:receiver.ne], receiver.Cn, receiver.Cf, receiver.Pn, XYZf_final, sp_init)
                xyz_final = vcat(xyz_final, XYZf_final)
            end


            msgout = Dict("Finished" => true,
                "Iter" => counter,
                "Loss" => Optim.minimum(res),
                "Q" => min[1:receiver.ne],
                "X" => xyz_final[:, 1],
                "Y" => xyz_final[:, 2],
                "Z" => xyz_final[:, 3],
                "Losstrace" => losses,
                "NodeTrace" => NodeTrace)


        HTTP.WebSockets.send(ws, json(msgout))

        catch error
            println(error)
        end
    end
end