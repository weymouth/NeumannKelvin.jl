# Panel collections wrapping a Table of `panels`, forwarding its columns and rows
abstract type PanelWrapper{T} <: AbstractVector{T} end
Base.getproperty(w::PanelWrapper,s::Symbol) = s in fieldnames(typeof(w)) ? getfield(w,s) : getproperty(getfield(w,:panels),s)
for f in [:size, :getindex, :setindex!]
    @eval @inline Base.@propagate_inbounds Base.$f(w::PanelWrapper, args...; kwargs...) = $f(w.panels, args...; kwargs...)
end

"""
    PanelTree(panels::Table;θ²=9)

Creates a tree structure from a set of `panels` that enables O(log N) evaluation of the 
panel influence instead of O(N). This is a huge speed-up for N≥O(100). The node monopole 
is used once `|x-bb|²/bb.R² > θ²` where `bb` is the node's bounding-box. Setting `θ²=Inf` 
would (inefficiently) evaluate all panels.

# Example
```julia
using NeumannKelvin
S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π, 0, 2π, hᵤ=1/50, N_max=3214) # quite a few panels
sys = PanelSystem(PanelTree(panels))          # accelerated body
gmressolve!(sys)                             # accelerated solve
extrema(cₚ(sys))                             # accelerated measurement                
```
"""
struct PanelTree{T, P<:Table, B, N, D} <: PanelWrapper{T}
    panels::P
    bvh::B
    nodes::N
    θ²::D
    PanelTree(panels::P, bvh::B, nodes::N, θ²::D) where {P<:Table,B,N,D} = 
        new{eltype(panels), P, B, N, D}(panels, bvh, nodes, θ²)
end
function PanelTree(panels;θ²=9)
    panels = Table(panels;q=zeros_like(panels.dA))
    bvh = BVH(map(BoundingVolume,panels.verts))
    PanelTree(panels,bvh,fill_nodes(panels,bvh),θ²)
end
struct MonoKernel <: GreenKernel end
function fill_nodes(panels,bvh)  # nodes only have monopole properties
    dA = aggregate(panels.dA,bvh)
    x = aggregate(panels.dA .* panels.x, bvh) ./ dA
    n = normalize.(aggregate(panels.dA .* panels.n, bvh))
    Table(;x,n,dA,q=zeros_like(dA),kernel=fill(MonoKernel(),length(dA)))
end

# Overload printing
Base.show(io::IO, pt::PanelTree) = println(io, "PanelTree($(length(pt.panels)) panels, $(pt.bvh.tree.levels) levels, θ²: $(pt.θ²))")
function Base.show(io::IO, ::MIME"text/plain", pt::PanelTree)
    show(io,pt)
    println(io, "  bounds: $(pt.bvh.nodes[1].lo) to $(pt.bvh.nodes[1].up)")
    println(io, "  panel type: $(eltype(pt.kernel))")
end

# Overload the two key functions, setting q and evaluating Φ!!
@inline function set_q!(BH::PanelTree,q)
    BH.q .= q .* BH.dA # set Q = q*dA on leaves
    aggregate!(BH.nodes.q,BH.q,BH.bvh) # aggregate
    BH.nodes.q ./= BH.nodes.dA # scale back to q
    BH.q .= q; BH     # reset leaf values
end
@inline Φ_boundary(x,(;panels,nodes,bvh,θ²)::PanelTree,(;mirrors)::Unbounded) = # tree-sum the free-space Green function
    sum(m->treesum((x,p)->p.q*∫G(x,p),x .* m,bvh,nodes,panels;val=zero(eltype(x)),θ²),mirrors)