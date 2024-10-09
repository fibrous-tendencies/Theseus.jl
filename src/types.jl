struct Objective
    ID::Int64
    W::Float64
    Indices::Vector{Int64}
    Values::AbstractMatrix{Float64}  # Using abstract type for flexibility

    function Objective(obj::Dict{String, Any}, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
        id = obj["OBJID"]
        w = Float64(obj["Weight"])

        # Determine indices
        if obj["Indices"][1] == -1
            indices = id == 1 ? N : (id == 0 ? F : collect(1:ne))
        else
            indices = Int64.(obj["Indices"]) .+ 1
        end

        # Determine values
        values = if haskey(obj, "Values")
            if obj["Indices"][1] == -1
                ones(ne) .* Float64.(obj["Values"])
            else
                Float64.(obj["Values"])
            end
        elseif haskey(obj, "Points")
            convert(Matrix{Float64}, reduce(hcat, obj["Points"])')
        else
            zeros(ne, 1)  # Using a zero matrix instead of `nothing` for type stability
        end

        new(id, w, indices, values)
    end
end

struct Parameters
    #OBJECTIVES
    Objectives::Vector{Objective}

    #PARAMETERS
    AbsTol::Float64
    RelTol::Float64
    Freq::Int64
    MaxIter::Int64
    Show::Bool

    #DERIVED VALUES
    UB::Vector{Float64}
    LB::Vector{Float64}

    NodeTrace::Bool

    function Objectives(objs::Vector{Any}, ne, nn, N, F)
        objectives = [Objective(obj, ne, nn, N, F) for obj in objs]
        return objectives
    end

    function Parameters(parameters::Dict{String, Any}, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
        objectives = [Objective(obj, ne, nn, N, F) for obj in parameters["Objectives"]]
        abstol = Float64(parameters["AbsTol"])
        reltol = Float64(parameters["RelTol"])
        freq = Int64(parameters["UpdateFrequency"])
        maxiter = Int64(parameters["MaxIterations"])
        show = Bool(parameters["ShowIterations"])

        ub = fill(Float64(parameters["UpperBound"]), ne)
        lb = fill(Float64(parameters["LowerBound"]), ne)

        nodeTrace = Bool(parameters["NodeTrace"])

        new(
            objectives,
            abstol,
            reltol,
            freq,
            maxiter,
            show,
            ub,
            lb,
            nodeTrace
        )
    end
end

struct AnchorParameters
    VAI::Vector{Int64}
    FAI::Vector{Int64}
    Init::Vector{Float64}

    function AnchorParameters(anchors::Vector{Dict{String, Any}}, var::Vector{Int}, fix::Vector{Int})
        variableAnchors = Int64.(var) .+ 1
        fixedAnchors = Int64.(fix) .+ 1

        init = Float64[]
        for anchor in anchors
            push!(init, anchor["InitialX"], anchor["InitialY"], anchor["InitialZ"])
        end

        new(variableAnchors, fixedAnchors, init)
    end
end

using SparseArrays

struct Receiver
    # FORCE DENSITY
    Q::Vector{Float64}

    # NETWORK INFORMATION
    N::Vector{Int64}
    F::Vector{Int64}

    XYZf::Matrix{Float64}
    Pn::Matrix{Float64}

    C::SparseMatrixCSC{Int64, Int64}
    Cn::SparseMatrixCSC{Int64, Int64}
    Cf::SparseMatrixCSC{Int64, Int64}

    ne::Int64
    nn::Int64

    Params::Union{Parameters, Nothing}
    AnchorParams::Union{AnchorParameters, Nothing}

    function Receiver(problem::Dict{String, Any})
        # Anchor Geometry
        xyzf = convert(Matrix{Float64}, reduce(hcat, problem["XYZf"])')

        # Global Info
        ne = Int64(problem["Network"]["Graph"]["Ne"])
        nn = Int64(problem["Network"]["Graph"]["Nn"])

        # Initial Force Densities
        q = length(problem["Q"]) == 1 ? fill(Float64(problem["Q"][1]), ne) : Float64.(problem["Q"])
        q = length(q) == ne ? q : fill(1.0, ne)  # Ensure length matches `ne`

        # Free/Fixed Nodes
        N = collect(1:length(problem["Network"]["FreeNodes"]))
        F = collect(length(N)+1:length(problem["Network"]["FixedNodes"]) + length(N))

        # Loads
        GH_p = convert(Matrix{Float64}, reduce(hcat, problem["P"])')
        p = zeros(nn, 3)

        if haskey(problem, "LoadNodes")
            LN = Int64.(problem["LoadNodes"]) .+ 1
            if length(LN) == size(GH_p, 1)
                p[LN, :] = GH_p
            elseif length(LN) > 1 && size(GH_p, 1) == 1
                p[LN, :] .= GH_p[1, :]
            else
                @warn "Number of load vectors must match number of load nodes or be 1. Using zero loads."
            end
            Pn = p[N, :]
        else
            if size(GH_p, 1) == 1
                p .= GH_p[1, :]
            elseif size(GH_p, 1) == length(N)
                p .= GH_p
            else
                @warn "Number of load vectors must match number of load nodes or be 1. Using zero loads."
            end
            Pn = p
        end

        # Connectivity
        i = Int64.(problem["I"]) .+ 1
        j = Int64.(problem["J"]) .+ 1
        v = Int64.(problem["V"])

        C = sparse(i, j, v, ne, nn)
        Cn = C[:, 1:length(N)]
        Cf = C[:, length(N)+1:end]

        # Parameters
        Params = haskey(problem, "Parameters") ? Parameters(problem["Parameters"], ne, nn, N, F) : nothing
        if Params !== nothing
            q .= clamp.(q, Params.LB, Params.UB)
        else
            @info "No optimization parameters provided. Running FDM using given force densities only."
        end

        # Anchor Parameters
        AnchorParams = haskey(problem, "VariableAnchors") ? 
            AnchorParameters(problem["VariableAnchors"], problem["NodeIndex"], problem["FixedAnchorIndices"]) : 
            nothing

        if AnchorParams === nothing
            @info "No anchor parameters provided. Using only given fixed node positions."
        end

        new(
            q, 
            N, 
            F,
            xyzf, 
            Pn,
            C,
            Cn,
            Cf,
            ne,
            nn,
            Params,
            AnchorParams
        )
    end
end