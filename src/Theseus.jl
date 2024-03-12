module Ariadne

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using Zygote
using Zygote: @adjoint
using HTTP
using JSON
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

export FDMsolve!

end # module FDMremote
