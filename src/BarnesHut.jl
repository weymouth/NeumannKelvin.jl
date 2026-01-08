"""
    BarnesHut(panels::Table;ϕ,sym_axes,kwargs...)

Creates a Barnes-Hut tree structure from a set of `panels` that enables O(log N)
evaluation of the panel influence instead of O(N). This is a huge speed-up for N>O(10).
**Note** System should be solved using a matrix-free method like `gmresSolve!`.

# All the `PanelSystem` fields _plus_
- `nodes::Table`: Aggregated node values in the binary tree
- `bvh::BVH`: Bounding volume hierarchy of the panels and nodes
- `d²=4`: Barnes-Hut distance cutoff. The node monopole is used once `|x-bb|²/bb.R² > d²`
where `bb` is the node's Bounding-Box. Setting `d²=Inf` would evaluate all N panels.

# Example
```julia
using NeumannKelvin
S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π, 0, 2π, hᵤ=1/50, N_max=3214) # quite a few panels
BH = BarnesHut(panels); gmressolve!(BH)
```

See also: [`PanelSystem`](@ref)
"""
struct BarnesHut{P<:PanelSystem, B, N} <: AbstractPanelSystem
    sys::P
    bvh::B
    nodes::N
end
function BarnesHut(sys::PanelSystem)
    bvh = BVH(BoundingVolume.(sys.panels))
    BarnesHut(sys,bvh,fill_nodes(sys.panels,bvh))
end
function fill_nodes(panels,bvh)  # nodes only have monopole properties
    dA = accumulate(panels.dA,bvh)
    x = accumulate(panels.dA .* panels.x, bvh) ./ dA
    n = normalize.(accumulate(panels.dA .* panels.n, bvh))
    add_columns(Table(;x,dA,n),q = 0.)
end
BarnesHut(args...;kwargs...) = BarnesHut(PanelSystem(args...;kwargs...))

# Overload properties
Base.getproperty(f::BarnesHut, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getfield(f.sys, s)
Base.setproperty!(f::BarnesHut, s::Symbol, x) = s in propertynames(f) ? setproperty!(f,s,x) : setproperty!(f.sys,s,x)

# Overload printing
Base.show(io::IO, BH::BarnesHut) = print(io, "BarnesHut($(length(BH.panels)) panels, $(BH.bvh.tree.levels) levels)")
function Base.show(io::IO, ::MIME"text/plain", BH::BarnesHut)
    abstract_show(io,BH); println(io, "  bounds: $(BH.bvh.nodes[1].lo) to $(BH.bvh.nodes[1].up)")
end

# Overload the two key functions, setting q and evaluating Φ!!
@inline function set_q!(BH::BarnesHut,q)
    BH.panels.q .= q
    accumulate!(BH.nodes.q,BH.panels.dA .* q,BH.bvh); BH.nodes.q ./= BH.nodes.dA
    BH
end
@inline Φ_sys(x,(;panels,nodes,bvh,kwargs)::BarnesHut;args...) = evaluate((x,p)->p.q*∫G(x,p;kwargs...),x,bvh,nodes,panels;kwargs...,args...)
