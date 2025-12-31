"""
    BarnesHut(panels::Table;ϕ,kwargs...)

Creates a Barnes-Hut tree structure from a set of `panels` that enables O(log N)
evaluation of the panel influence instead of O(N). This is a huge speed-up for N>O(10).

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
BH = BarnesHut(panels); GMRESsolve!(BH) # you can also do BarnesHutsolve(panels)
```

See also: [`PanelSystem`](@ref)
"""
struct BarnesHut{TP,TN,TB,F,KW} <: AbstractPanelSystem
    panels::TP
    nodes::TN
    bvh::TB
    ϕ::F
    kwargs::KW
end
function BarnesHut(panels;ϕ=∫G,kwargs...)
    bvh = bvh_panels(panels)
    nodes = fill_nodes(panels,bvh)
    q = similar(panels.dA); q .= 0
    BarnesHut(Table(panels;q),Table(nodes,q=similar(nodes.dA)),bvh,ϕ,kwargs)
end

# Pretty printing
Base.show(io::IO, BH::BarnesHut) = print(io, "BarnesHut($(length(BH.panels)) panels, $(BH.bvh.tree.levels) levels)")
function Base.show(io::IO, ::MIME"text/plain", BH::BarnesHut)
    abstract_show(io,BH); println(io, "  bounds: $(BH.bvh.nodes[1].lo) to $(BH.bvh.nodes[1].up)")
end

# Overload a few functions
total_area(BH::BarnesHut) = BH.nodes[1].dA
@inline function set_q!(BH::BarnesHut,q)
    BH.panels.q .= q
    accumulate!(BH.nodes.q,BH.panels.dA .* q,BH.bvh); BH.nodes.q ./= BH.nodes.dA
    BH
end

"""
    Φ(x,BH::BarnesHut)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')da' = ∑ᵢqᵢϕ(x,pᵢ)` induced by **solved** panel system `BH`.
The Barnes-Hut approximation uses the aggregated node information in the tree to avoid the sum
over all N panels.
"""
@inline Φ(x,(;panels,nodes,bvh,ϕ,kwargs)::BarnesHut;args...) = evaluate((x,p)->p.q*ϕ(x,p,kwargs...),x,bvh,nodes,panels;kwargs...,args...)

"""
    BarnesHutsolve(panels::Table,b;...) = GMRESsolve!(BarnesHut(panels;...),b;...)

See: [`BarnesHut`](@ref), [`GMRESSolve!`](@ref)
"""
BarnesHutsolve(panels,b=components(panels.n,1);atol=1e-3,kwargs...) = GMRESsolve!(BarnesHut(panels;kwargs...),b;atol)
