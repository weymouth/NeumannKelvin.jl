"""
    BarnesHut(panels::Table;d²=4)

Creates a Barnes-Hut tree structure from a set of `panels` that enables O(log N)
evaluation of the panel influence instead of O(N). This is a huge speed-up for N≥O(100).
The node monopole is used once `|x-bb|²/bb.R² > d²` where `bb` is the node's Bounding-Box. 
Setting `d²=Inf` would (inefficiently) evaluate all panels.

# Example
```julia
using NeumannKelvin
S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π, 0, 2π, hᵤ=1/50, N_max=3214) # quite a few panels
BH = BodyPanelSystem(BarnesHut(panels)); gmressolve!(BH)
```
"""
struct BarnesHut{T, P<:Table, B, N, D} <: AbstractVector{T}
    panels::P
    bvh::B
    nodes::N
    d²::D
    BarnesHut(panels::P, bvh::B, nodes::N, d²::D) where {P<:Table,B,N,D} = 
        new{eltype(panels), P, B, N, D}(panels, bvh, nodes, d²)
end
function BarnesHut(panels;d²=4)
    bvh = BVH(map(BoundingVolume,panels))
    BarnesHut(panels,bvh,fill_nodes(panels,bvh),d²)
end
struct MonoKernel <: GreenKernel end
function fill_nodes(panels,bvh)  # nodes only have monopole properties
    dA = accumulate(panels.dA,bvh)
    x = accumulate(panels.dA .* panels.x, bvh) ./ dA
    n = normalize.(accumulate(panels.dA .* panels.n, bvh))
    Table(;x,n,dA,q=zeros_like(dA),kernel=fill(MonoKernel(),length(dA)))
end

# Overload properties
Base.getproperty(f::BarnesHut, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getproperty(f.panels, s)
Base.setproperty!(f::BarnesHut, s::Symbol, x) = s in propertynames(f) ? error("BarnesHut is immutable") : setproperty!(f.panels,s,x)
for f in [:size, :getindex, :setindex!]
    @eval @inline Base.@propagate_inbounds Base.$f(bh::BarnesHut, args...; kwargs...) = $f(bh.panels, args...; kwargs...)
end

# Overload printing
Base.show(io::IO, BH::BarnesHut) = print(io, "BarnesHut($(length(BH.panels)) panels, $(BH.bvh.tree.levels) levels, d²: $(BH.d²))")
function Base.show(io::IO, ::MIME"text/plain", BH::BarnesHut)
    show(io,BH); println(io,"")
    println(io, "  bounds: $(BH.bvh.nodes[1].lo) to $(BH.bvh.nodes[1].up)")
    println(io, "Panel type: $(eltype(BH.kernel))")
end

# Overload the two key functions, setting q and evaluating Φ!!
@inline function set_q!(BH::BarnesHut,q)
    BH.q .= q .* BH.dA # set Q = q*dA on leaves
    accumulate!(BH.nodes.q,BH.q,BH.bvh) # accumulate
    BH.nodes.q ./= BH.nodes.dA # scale back to q
    BH.q .= q; BH     # reset leaf values
end
@inline Φ_dom(x,(;panels,nodes,bvh,d²)::BarnesHut) = evaluate((x,p)->p.q*∫G(x,p),x,bvh,nodes,panels;val=zero(eltype(x)),d²)
