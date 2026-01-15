"""
    PanelTree(panels::Table;θ²=4)

Creates a tree structure from a set of `panels` that enables O(log N) evaluation of the 
panel influence instead of O(N). This is a huge speed-up for N≥O(100).
The node monopole is used once `|x-bb|²/bb.R² > θ²` where `bb` is the node's Bounding-Box. 
Setting `θ²=Inf` would (inefficiently) evaluate all panels.

# Example
```julia
using NeumannKelvin
S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π, 0, 2π, hᵤ=1/50, N_max=3214) # quite a few panels
sys = BodyPanelSystem(panels,wrap=PanelTree) # build System based on PanelTree
gmressolve!(sys)                             # accelerated solve
extrema(cₚ(sys))                             # accelerated measurement                
```
"""
struct PanelTree{T, P<:Table, B, N, D} <: AbstractVector{T}
    panels::P
    bvh::B
    nodes::N
    θ²::D
    PanelTree(panels::P, bvh::B, nodes::N, θ²::D) where {P<:Table,B,N,D} = 
        new{eltype(panels), P, B, N, D}(panels, bvh, nodes, θ²)
end
function PanelTree(panels;θ²=4)
    bvh = BVH(map(BoundingVolume,panels))
    PanelTree(panels,bvh,fill_nodes(panels,bvh),θ²)
end
struct MonoKernel <: GreenKernel end
function fill_nodes(panels,bvh)  # nodes only have monopole properties
    dA = aggregate(panels.dA,bvh)
    x = aggregate(panels.dA .* panels.x, bvh) ./ dA
    n = normalize.(aggregate(panels.dA .* panels.n, bvh))
    Table(;x,n,dA,q=zeros_like(dA),kernel=fill(MonoKernel(),length(dA)))
end

# Overload properties
Base.getproperty(f::PanelTree, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getproperty(f.panels, s)
Base.setproperty!(f::PanelTree, s::Symbol, x) = s in propertynames(f) ? error("PanelTree is immutable") : setproperty!(f.panels,s,x)
for f in [:size, :getindex, :setindex!]
    @eval @inline Base.@propagate_inbounds Base.$f(bh::PanelTree, args...; kwargs...) = $f(bh.panels, args...; kwargs...)
end

# Overload printing
Base.show(io::IO, pt::PanelTree) = print(io, "PanelTree($(length(BH.panels)) panels, $(BH.bvh.tree.levels) levels, θ²: $(BH.θ²))")
function Base.show(io::IO, ::MIME"text/plain", BH::PanelTree)
    show(io,BH); println(io,"")
    println(io, "  bounds: $(BH.bvh.nodes[1].lo) to $(BH.bvh.nodes[1].up)")
    println(io, "Panel type: $(eltype(BH.kernel))")
end

# Overload the two key functions, setting q and evaluating Φ!!
@inline function set_q!(BH::PanelTree,q)
    BH.q .= q .* BH.dA # set Q = q*dA on leaves
    aggregate!(BH.nodes.q,BH.q,BH.bvh) # aggregate
    BH.nodes.q ./= BH.nodes.dA # scale back to q
    BH.q .= q; BH     # reset leaf values
end
@inline Φ_dom(x,(;panels,nodes,bvh,θ²)::PanelTree) = treesum((x,p)->p.q*∫G(x,p),x,bvh,nodes,panels;val=zero(eltype(x)),θ²)
