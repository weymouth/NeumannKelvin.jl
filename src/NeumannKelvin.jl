module NeumannKelvin

# Useful datatypes & functions
using Reexport
@reexport using TypedTables,StaticArrays
Base.adjoint(t::Table) = permutedims(t)
@reexport using ForwardDiff: derivative,gradient
@reexport using LinearAlgebra: ×,⋅,tr

# Quadrature functions & utilities
include("quad.jl")

# Green function definitions
using ThreadsX # multi-threaded map,sum, & foreach
include("green.jl")
export source,kelvin

# Panel method
include("panel_method.jl")
export param_props,ϕ,∂ₙϕ,Uₙ,influence,influence!,φ,∇φ,ζ,steady_force,added_mass

end
