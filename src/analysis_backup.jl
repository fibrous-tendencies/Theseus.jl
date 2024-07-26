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
            #try
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
                        if receiver.Params.NodeTrace == true
                            push!(NodeTrace, deepcopy(xyzfull))
                        end

                        i += 1

                        if receiver.Params.Show && i % receiver.Params.Freq == 0
                            
                            push!(iters, Q)
                            push!(losses, loss)


                            if receiver.Params.NodeTrace == true
                                #send intermediate message
                                msgout = Dict("Finished" => false,
                                    "Iter" => i, 
                                    "Loss" => loss,
                                    "Q" => Q, 
                                    "X" => last(NodeTrace)[:,1], 
                                    "Y" => last(NodeTrace)[:,2], 
                                    "Z" => last(NodeTrace)[:,3],
                                    "Losstrace" => losses)
                            else
                                msgout = Dict("Finished" => false,
                                    "Iter" => i, 
                                    "Loss" => loss,
                                    "Q" => Q, 
                                    "X" => xyzfull[:,1], 
                                    "Y" => xyzfull[:,2], 
                                    "Z" => xyzfull[:,3],
                                    "Losstrace" => losses)
                            end
                                
                            HTTP.WebSockets.send(ws, json(msgout))
                        end
                    end
                end
                
                return loss
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



            function wolfe_conditions(f, g, x, d, alpha; c1=1e-4, c2=0.9)
                # Wolfe conditions
                f_x = f(x)
                g_x = g(x)
                f_new = f(x + alpha * d)
                g_new = g(x + alpha * d)
                
                # First Wolfe condition: Armijo condition
                armijo = f_new ≤ f_x + c1 * alpha * dot(g_x, d)
                
                # Second Wolfe condition: Curvature condition
                curvature = abs(dot(g_new, d)) ≤ c2 * abs(dot(g_x, d))
                
                return armijo && curvature
            end
            
            function gd_line_search(f, g, x, d, alpha0; max_iter=1000, min_val=1e-8, tol=1e-6, c1=1e-4, c2=0.9)
                alpha = alpha0
                best_alpha = alpha
                best_f_val = f(x)
                
                for _ in 1:max_iter
                    # Compute the new point and function value
                    x_new .= x + alpha * d
                    if any(x_new .<= min_val)
                        alpha ./= 2.0
                        best_alpha .= alpha
                    else
                        if wolfe_conditions(f, g, x_new, d, alpha;)
                            return best_alpha, best_f_val, x_new
                        else
                            alpha ./= 2.0
                        end
                    end
                end
                
                # Final update with the best step size found
                if any(x_new .<= min_val)
                    x_new = x 
                    best_f_val = f(x)
                else
                    x_new = x + best_alpha * d
                    best_f_val = f(x_new)
                end

                return best_alpha, best_f_val, x_new
            end            

            # Define the Gradient Descent optimization function with custom line search
            function gd_simple(f, x0; max_iter=10, tol=1e-6, min_val=1e-8)
                x .= max.(x0, min_val) # prevent initially negative force densities
                fval .= f(x) # initial loss value with input parameters
                g = x -> Zygote.gradient(f, x)[1] # gradient function
                gval .= g(x) # initial gradient value
                k = 0 # iteration counter
                d = zeros(length(x0))
                
                while norm(gval) > tol && k < max_iter

                    gval .= g(x) # gradient value

                    d .= -gval ./ maximum(abs.(gval)) # search direction
                    # Custom line search to ensure non-negative parameter values
                    best_alpha, fval_new, x_new = gd_line_search(f, g, x, d, 10.0; min_val=min_val)

                    #If the change in loss value is small, just return
                    if k > 0 && abs(fval - fval_new) < tol
                        return x, fval, norm(gval) <= tol, k
                    end

                    x .= x_new # normalize and update parameters
                    fval .= fval_new #update fval

                    k += 1 #increment counter

                end
                
                return x, fval, norm(gval) <= tol, k
            end

            res = gd_simple(obj, parameters, max_iter = receiver.Params.MaxIter)
            #res = lbfgs_custom(obj, parameters, max_iter = receiver.Params.MaxIter)

            println("Result :", res)

            min = res[1]


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
                "Iter" => res[4],
                "Loss" => res[2],
                "Q" => min[1:receiver.ne],
                "X" => xyz_final[:, 1],
                "Y" => xyz_final[:, 2],
                "Z" => xyz_final[:, 3],
                "Losstrace" => losses,
                "NodeTrace" => NodeTrace)


        HTTP.WebSockets.send(ws, json(msgout))

        #catch error
        #    println(error)
        #end
    end
end