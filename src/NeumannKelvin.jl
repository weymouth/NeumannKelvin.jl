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
include("kelvin.jl")
export ∫kelvin

# Panels
include("panels.jl")
export param_props,panelize

# Panel method
include("panel_method.jl")
export ∂ₙϕ,influence,ζ,steady_force,added_mass

components(data) = ntuple(i -> getindex.(data, i), 3)
export components

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008" include("../ext/NeumannKelvinWGLMakieExt.jl")
    end
end
end
