module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using Zygote
using Zygote: @adjoint
using JSON
using HTTP
using LineSearches
using ChainRulesCore

include("FDM.jl")
include("types.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("adjoint.jl")
include("anchors.jl")

export start!

end # module FDMremote
