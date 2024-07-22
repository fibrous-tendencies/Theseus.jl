#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(
    q::Vector{Float64}, #Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64}, #Index matrix of fixed nodes
    Pn::Matrix{Float64}, #Matrix of free node loads
    Nf::Matrix{Float64}, #Matrix of fixed node positions
    sp_init::Vector{Int64} #Intialization for sparse matrix
    )

    Q = sparse(sp_init,sp_init,q) # build diagonal force density Matrix
    
    return (Cn' * Q * Cn) \ (Pn - Cn' * Q * Cf * Nf)
end



function optimize_FDM(
    q::Vector{Float64}, 
    grad::Vector{Float64}, 
    iter::{Int64},
    rel_tol::{Float64},
    abs_tol::{Float64},
     )

     xyz = nothing

     return xyz
end

```
Constrained line search to ensure sign never flips 
````
function linesearch_FDM()
end



