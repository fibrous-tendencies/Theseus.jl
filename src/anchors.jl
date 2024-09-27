"""
Returns a sorted matrix of points based on their indices in the original matrix.
This allows for changes to some elements of the matrix in a differentiable way since zygote does not support
mutating operations and the typical assignment operator mutates.
"""
function combineSorted(xyzNew::Matrix{Float64}, xyz::Matrix{Float64}, changedIndex::Vector{Int64}, unchangedIndex::Vector{Int64})
    total_length = length(changedIndex) + length(unchangedIndex)
    xyzunsorted = Matrix{Float64}(undef, total_length, size(xyz, 2))
    xyzunsorted[1:length(changedIndex), :] = xyzNew
    xyzunsorted[length(changedIndex)+1:end, :] = xyz
    indices = vcat(changedIndex, unchangedIndex)
    sort_order = sortperm(indices)
    return xyzunsorted[sort_order, :]
end