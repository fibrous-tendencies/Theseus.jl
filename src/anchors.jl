function newAnchor()
end


"""
Returns a sorted matrix of points based on their indices in the original matrix.
This allows for changes to some elements of the matrix in a differentiable way since zygote does not support
mutating operations and the typical assignment operator mutates.
"""
function combineSorted(xyzNew::Matrix{Float64}, xyz::Matrix{Float64}, changedIndex::Vector{Int64}, unchangedIndex::Vector{Int64})
    xyzunsorted = [xyzNew; xyz]
    i = sortperm([changedIndex; unchangedIndex])

    return xyzunsorted[i, :]
end
