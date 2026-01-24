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
export source,∫G,Φ,∇Φ,u,cₚ,steadyforce,addedmass

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
export NKPanelSystem,∫NK,kelvin

# General support functions
components(data,i) = getindex.(data, i)
components(data::AbstractArray{S}) where {S<:SVector{n}} where n = ntuple(i->components(data,i),n)
extent(a) = (p = extrema(a); p[2]-p[1])
"""
    viz(panels::Table, values=panels.dA; vectors=panels.n, kwargs...)
    viz(sys::BodyPanelSystem, values=cₚ(sys); vectors=u(sys), kwargs...)
    viz(sys::FSPanelSystem, values=cₚ(sys); vectors=u(sys), freesurf=true, kwargs...)
    viz(sys::NKPanelSystem, values=cₚ(sys); vectors=u(sys), freesurf=true, 
        fsargs=(;s=1/2+2πℓ,h=ℓ/3,half=true), kwargs...)

Vizualizes a table of `panels` or a `PanelSystem` in 3D. Panels are colored by the `value` array
and the `vectors` are plotted extending from each panel center. If `freesurf` is true (default) for
a free-surface panel system, the free surface is also plotted. The `fsargs` keyword tuple allows 
passing arguments to define the free surface extents and resolution for `NKPanelSystem`s.

** Note ** you must load a Makie backend (GLMakie or WGLMakie) to use this function.

# Example
    using NeumannKelvin,WGLMakie
    panels = panelize((u,v)->SA[cos(u),cos(v)*sin(u),sin(v)*sin(u)-1.5],0,pi,0,2pi)
    viz(panels,vscale=3) # view panels, normals scaled by 3

    sys = NKPanelSystem(panels,ℓ=1/4) |> directsolve!
    viz(sys;fsargs=(s=2)) # view NK solution with free surface extent s=2
"""
viz(args...; kwargs...) = @warn "Load GLMakie or WGLMakie for viz functionality."
function cₚu(sys)
    vectors = u(sys); U² = sum(abs2,sys.U); cp = @. 1-sum(abs2,vectors)/U²
    return cp,vectors
end
function xyζ(sys::NKPanelSystem;s=1/2+2π*sys.args.ℓ,h=sys.args.ℓ/3,half=true)
    x,y=-2s:h:s,ifelse(half,0,-s):h:s
    return x,y,ζ(x,y,sys) .* sys.args.ℓ
end
function xyζ(sys::FSPanelSystem; ignore...)
    x,y,_ = reshape.(components(sys.freesurf.x),Ref(size(sys.fsm)))
    return x,y,ζ(sys) .* sys.ℓ
end
export viz,components,extent

# Initialize chebregions on precompile instead of when `using NeumannKelvin`
__init__() = chebregions[] = (makecheb(eps(),1),makecheb(1,4),makecheb(4,10),makecheb(1e-5,1;xfrm=r2R))
end