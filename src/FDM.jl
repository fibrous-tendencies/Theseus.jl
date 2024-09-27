#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(
    q::Vector{Float64}, #Vector of force densities
    Cn::SparseMatrixCSC{Float64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Float64,Int64}, #Index matrix of fixed nodes
    Pn::Matrix{Float64}, #Matrix of free node loads
    Nf::Matrix{Float64} #Matrix of fixed node positions
    )
    # q is a vector of size (ne,)

    # Multiply each row of Cn by corresponding q element
    QCn = Cn .* q  # Element-wise multiplication
    QCf = Cf .* q  # Element-wise multiplication
    A = transpose(Cn) * QCn
    b = Pn - transpose(Cn) * (QCf * Nf)
    x = A \ b
    return x
end


