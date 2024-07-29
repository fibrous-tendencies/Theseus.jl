import HTTP.WebSockets

function FDMoptim!(receiver, ws)
    sp_init = collect(Int64, range(1, length = receiver.ne))

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
        println("OPTIMIZING")

        Q = Vector{Float64}(undef, receiver.ne)
        NodeTrace = []

        i = 0
        iters = Vector{Vector{Float64}}()
        losses = Vector{Float64}()

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
                    Q .= q
                    if receiver.Params.NodeTrace == true
                        push!(NodeTrace, copy(xyzfull))
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
                    Q .= q
                    if receiver.Params.NodeTrace == true
                        push!(NodeTrace, copy(xyzfull))
                    end

                    i += 1

                    if receiver.Params.Show && i % receiver.Params.Freq == 0
                        push!(iters, copy(Q))
                        push!(losses, loss)

                        if receiver.Params.NodeTrace == true
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

        objective = if isnothing(receiver.AnchorParams)
            obj
        else
            obj_xyz
        end

        parameters = if isnothing(receiver.AnchorParams)
            receiver.Q
        else
            vcat(receiver.Q, receiver.AnchorParams.Init)
        end

        function wolfe_conditions(f, g, x, d, alpha; c1=1e-4, c2=0.9)
            f_x = f(x)
            g_x = g(x)
            f_new = f(x + alpha * d)
            g_new = g(x + alpha * d)

            armijo = f_new ≤ f_x + c1 * alpha * dot(g_x, d)
            curvature = abs(dot(g_new, d)) ≤ c2 * abs(dot(g_x, d))

            return armijo && curvature
        end

        function gd_line_search(f, g, x, d, alpha0::Float64; max_iter=100, min_val=1e-8, min_threshold=1e-4, tol=1e-6, c1=1e-4, c2=0.9)
            alpha = alpha0
            best_f_val = f(x)
            x_test = x

            for _ in 1:max_iter
                x_test = x .+ alpha .* d
                #If any values are made less than the minimum value by d
                #Set direction for those to 0

                if any(x_test .<= min_val)

                    d_0 = d

                    for (i, test_dir) in enumerate(x_test)
                        if test_dir <= min_val
                            d_0[i] = 0
                        end
                    end

                    x_test = x .+ alpha .* d_0
                    best_f_val = f(x_test)

                    if wolfe_conditions(f, g, x_test, d_0, alpha)
                        println("Masked Wolfe conditions met, alpha: ")
                        return alpha, best_f_val, x_test
                    else
                        alpha /= 2.0
                    end
                else
                    best_f_val = f(x_test)
                    if wolfe_conditions(f, g, x_test, d, alpha)
                        return alpha, best_f_val, x_test
                    else
                        alpha /= 2.0
                    end
                end
            end

            return alpha, best_f_val, x_test
        end

        function gd_simple(f, x0::Vector{Float64}; max_iter::Int64 = 10, tol::Float64 = 1e-6, min_val::Float64 = 1e-8)
            x = max.(x0, min_val)
            fval = f(x)
            g = x -> Zygote.gradient(f, x)[1]
            gval = g(x)
            k = 0

            while norm(gval) > tol && k < max_iter
                gval = g(x)
                d = -gval / maximum(abs.(gval))
                best_alpha, fval_new, x_new = gd_line_search(f, g, x, d, 10.0; min_val=min_val)

                if k > 0 && abs(fval - fval_new) < tol
                    return x, fval, norm(gval) <= tol, k
                end

                x = x_new
                fval = fval_new

                k += 1
            end

            return x, fval, norm(gval) <= tol, k
        end

        res = gd_simple(objective, parameters, max_iter = receiver.Params.MaxIter)

        

        min = res[1]

        println("------------------------------------")

        println("SOLUTION FOUND")
        println("Objective Value: ", res[2])
        println("Converged :", res[3])
        println("Iterations :", res[4])

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
    end
end
