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
export measure,panelize

# BodyPanelSystem
include("BodyPanelSystem.jl")
export BodyPanelSystem,bodyarea,bodyvol

import AcceleratedKernels as AK # multi-threaded mapreduce,foreachindex
include("panel_method.jl")
export ∫G,∂ₙϕ,Φ,∇Φ,u,cₚ,steadyforce,addedmass

# Direct and matrix-free solver
include("solvers.jl")
export gmressolve!,directsolve!

# Barnes-Hut functions and PanelTree wrapper
include("BarnesHutCore.jl")
include("PanelTree.jl")
export PanelTree

# Free-surface panel system
include("FSPanelSystem.jl")
export FSPanelSystem,ζ

# Kelvin Green function definitions
include("NKPanelSystem.jl")
export NKPanelSystem,kelvin

# General support functions
components(data,i) = getindex.(data, i)
components(data::AbstractArray{S}) where {S<:SVector{n}} where n = ntuple(i->components(data,i),n)
extent(a) = (p = extrema(a); p[2]-p[1])
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
viz(sys::AbstractPanelSystem;vscale=1,kwargs...) = ((cp,vectors)=cₚu(sys);viz(sys.body,cp;vectors,vscale,label="cₚ",kwargs...))
viz(args...; kwargs...) = @warn "Load Plots or GLMakie (terminal/VSCode) or WGLMakie (browser/Pluto) for viz functionality."
function cₚu(sys)
    vectors = u(sys); U² = sum(abs2,sys.U); cp = @. 1-sum(abs2,vectors)/U²
    return cp,vectors
end
function xyζ(sys::NKPanelSystem,s=1/2+2π*sys.args.ℓ,h=sys.args.ℓ/3,half=true)
    x,y=-2s:h:s,ifelse(half,0,-s):h:s
    return x,y,ζ(x,y,sys) .* sys.args.ℓ
end
function xyζ(sys::FSPanelSystem)
    x,y,_ = reshape.(components(sys.freesurf.x),Ref(size(sys.fsm)))
    return x,y,ζ(sys) .* sys.ℓ
end
export viz,components,extent

# Initialize chebregions on precompile instead of when `using NeumannKelvin`
__init__() = chebregions[] = (makecheb(eps(),1),makecheb(1,4),makecheb(4,10),makecheb(1e-5,1;xfrm=r2R))
end