# types.jl

# Define the Objective struct with parametric types for improved type stability
struct Objective{TIndices<:AbstractVector{Int64}, TValues}
    ID::Int64
    W::Float64
    Indices::TIndices
    Values::TValues
end

# Constructor for Objective
function Objective(obj::Dict, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
    id = obj["OBJID"]
    w = Float64(obj["Weight"])

    # Ensure Indices is always a Vector{Int64}
    indices = if obj["Indices"][1] == -1
        if id == 1  # Target objective
            N
        elseif id == 0  # Fixed nodes
            F
        else  # Edge-based objective
            collect(1:ne)
        end
    else
        Int64.(obj["Indices"]) .+ 1  # Adjust for 1-based indexing in Julia
    end

    # Ensure Values is always a consistent type
    if haskey(obj, "Values")
        values_array = Float64.(obj["Values"])
        if length(values_array) == 1
            values = values_array[1]  # Store as scalar Float64
        else
            values = values_array  # Store as Vector{Float64}
        end
    elseif haskey(obj, "Points")
        # Convert list of points to Matrix{Float64}
        points = obj["Points"]
        values_matrix = convert(Matrix{Float64}, reduce(hcat, points)')  # Transpose after concatenation
        values = values_matrix
    else
        # Use an empty Vector{Float64} if no values are provided
        values = Float64[]
    end

    return Objective{typeof(indices), typeof(values)}(id, w, indices, values)
end

# Define the Parameters struct with parametric types
struct Parameters{TObjective}
    Objectives::Vector{TObjective}
    AbsTol::Float64
    RelTol::Float64
    Freq::Int64
    MaxIter::Int64
    Show::Bool
    UB::Vector{Float64}
    LB::Vector{Float64}
    NodeTrace::Bool
end

# Helper function to create Objectives vector
function createObjectives(objs::Vector{Any}, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
    objectives = Objective[]
    for obj in objs
        push!(objectives, Objective(obj, ne, nn, N, F))
    end
    return objectives
end

# Constructor for Parameters
function Parameters(parameters::Dict, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
    objectives = createObjectives(parameters["Objectives"], ne, nn, N, F)
    abstol = Float64(parameters["AbsTol"])
    reltol = Float64(parameters["RelTol"])
    freq = Int64(parameters["UpdateFrequency"])
    maxiter = Int64(parameters["MaxIterations"])
    show = Bool(parameters["ShowIterations"])
    ub = fill(Float64(parameters["UpperBound"]), ne)
    lb = fill(Float64(parameters["LowerBound"]), ne)
    nodetrace = Bool(parameters["NodeTrace"])

    # Get the concrete type of the Objective instances
    TObjective = typeof(objectives[1])

    return Parameters{TObjective}(
        objectives,
        abstol,
        reltol,
        freq,
        maxiter,
        show,
        ub,
        lb,
        nodetrace
    )
end

# Default Parameters constructor for cases where no parameters are provided
function DefaultParameters(ne::Int64)
    objectives = Objective[]
    abstol = 1e-6
    reltol = 1e-6
    freq = 1
    maxiter = 1000
    show = false
    ub = fill(Inf, ne)
    lb = fill(-Inf, ne)
    nodetrace = false

    return Parameters{Objective}(
        objectives,
        abstol,
        reltol,
        freq,
        maxiter,
        show,
        ub,
        lb,
        nodetrace
    )
end

# Define the AnchorParameters struct
struct AnchorParameters
    VAI::Vector{Int64}
    FAI::Vector{Int64}
    Init::Vector{Float64}
end

# Constructor for AnchorParameters
function AnchorParameters(anchors::Vector{Any}, var::Vector{Any}, fix::Vector{Any})
    variableAnchors = Int64.(var) .+ 1
    fixedAnchors = Int64.(fix) .+ 1

    init = Float64[]
    for anchor in anchors
        push!(init, Float64(anchor["InitialX"]))
        push!(init, Float64(anchor["InitialY"]))
        push!(init, Float64(anchor["InitialZ"]))
    end

    return AnchorParameters(
        variableAnchors,
        fixedAnchors,
        init
    )
end

# Default AnchorParameters constructor for cases where no anchor parameters are provided
function DefaultAnchorParameters()
    return AnchorParameters(Int64[], Int64[], Float64[])
end

# Define the Receiver struct with parametric types
struct Receiver{TParams, TAnchorParams}
    Q::Vector{Float64}
    N::Vector{Int64}
    F::Vector{Int64}
    XYZf::Matrix{Float64}
    Pn::Matrix{Float64}
    C::SparseMatrixCSC{Float64, Int64}
    Cn::SparseMatrixCSC{Float64, Int64}
    Cf::SparseMatrixCSC{Float64, Int64}
    ne::Int64
    nn::Int64
    Params::TParams
    AnchorParams::TAnchorParams
end

# Constructor for Receiver
function Receiver(problem::Dict)
    # Anchor geometry
    xyzf_list = problem["XYZf"]
    xyzf = convert(Matrix{Float64}, reduce(hcat, xyzf_list)')  # Transpose after concatenation

    # Global info
    ne = Int(problem["Network"]["Graph"]["Ne"])
    nn = Int(problem["Network"]["Graph"]["Nn"])

    # Initial force densities
    if length(problem["Q"]) == 1
        q = fill(Float64(problem["Q"][1]), ne)
    elseif length(problem["Q"]) == ne
        q = Float64.(problem["Q"])
    else
        q = fill(1.0, ne)
    end

    # Free and fixed nodes
    N_indices = Int.(problem["Network"]["FreeNodes"]) .+ 1
    F_indices = Int.(problem["Network"]["FixedNodes"]) .+ 1

    N = collect(1:length(N_indices))
    F = collect(length(N_indices)+1:length(N_indices)+length(F_indices))

    # Loads
    GH_p_list = problem["P"]
    GH_p = convert(Matrix{Float64}, reduce(hcat, GH_p_list)')  # Transpose after concatenation

    if haskey(problem, "LoadNodes")
        LN = Int.(problem["LoadNodes"]) .+ 1
        p = zeros(nn, 3)
        if length(LN) == size(GH_p, 1)
            p[LN, :] = GH_p
        elseif length(LN) > 1 && size(GH_p, 1) == 1
            p[LN, :] = repeat(GH_p, length(LN), 1)
        else
            println("Warning: Number of load vectors must match number of load nodes or be 1.")
            println("Using zero loads.")
        end
        Pn = p[N_indices, :]
    else
        if size(GH_p, 1) == 1
            p = repeat(GH_p, length(N_indices), 1)
        elseif size(GH_p, 1) == length(N_indices)
            p = GH_p
        else
            println("Warning: Number of load vectors must match number of free nodes or be 1.")
            println("Using zero loads.")
            p = zeros(length(N_indices), 3)
        end
        Pn = p
    end

    # Connectivity
    i = Int.(problem["I"]) .+ 1
    j = Int.(problem["J"]) .+ 1
    v = Float64.(problem["V"])

    C = sparse(i, j, v, ne, nn)
    Cn = @view C[:, N_indices]
    Cf = @view C[:, F_indices]

    # Handle Params
    if haskey(problem, "Parameters")
        params = Parameters(problem["Parameters"], ne, nn, N, F)
        # Prevent force densities from being outside the bounds
        q = clamp.(q, params.LB, params.UB)
    else
        println("No optimization parameters provided.")
        println("Running FDM using given force densities only.")
        params = DefaultParameters(ne)
    end

    # Handle AnchorParams
    if haskey(problem, "VariableAnchors")
        varAnchors = problem["NodeIndex"]
        fixAnchors = problem["FixedAnchorIndices"]
        anchorParams = AnchorParameters(problem["VariableAnchors"], varAnchors, fixAnchors)
    else
        println("No anchor parameters provided.")
        println("Using only given fixed node positions.")
        anchorParams = DefaultAnchorParameters()
    end

    # Get the concrete types
    TParams = typeof(params)
    TAnchorParams = typeof(anchorParams)

    return Receiver{TParams, TAnchorParams}(
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
        params,
        anchorParams
    )
end
