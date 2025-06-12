module NeumannKelvin

# Useful datatypes & functions
using Reexport
@reexport using TypedTables,StaticArrays
Base.adjoint(t::Table) = permutedims(t)
@reexport using ForwardDiff: derivative,gradient
@reexport using LinearAlgebra: ×,⋅,tr,norm

# Quadrature functions & utilities
include("quad.jl")

# Panel set-up
include("panels.jl")
export measurepanel,surfsplit,panelize

# Panel method
using ThreadsX # multi-threaded map,sum, & foreach
include("panel_method.jl")
export ∫G,∂ₙϕ,influence,ζ,steady_force,added_mass

# Kelvin Green function definitions
include("kelvin.jl")
export ∫kelvin,reflect

# Plotting fallback
"""
    viz(panels::Table, values=panels.dA; vectors=0.3panels.n, kwargs...)

Vizualizes a table of `panels` in 3D. Panels are colored by the `value` array
and the `vectors` are plotted extending from each panel center.

# Details

If Plots is loaded, the panels are visualized with colored markers and the
vectors are grey lines.

If a Makie library is loaded, the panels are visualized as two triangles extending
to the corners, and the vectors are colored 3D arrows.

# Example
    using NeumannKelvin,Plots
    panels = panelize((u,v)->SA[cos(u),cos(v)*sin(u),sin(v)*sin(u)],0,pi,0,2pi)
    viz(panels)
"""
viz(args...; kwargs...) = @warn "Pass a Table() of panels and load Plots or GLMakie (terminal/VSCode) or WGLMakie (browser/Pluto) for viz functionality."
components(data::AbstractVector{T}) where T<:SVector{n} where n = ntuple(i -> getindex.(data, i), n)
export viz,components

end