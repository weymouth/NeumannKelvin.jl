"""
    BarnesHut(panels::Table;ϕ,kwargs...) -> BarnesHut

Creates a BarnesHut tree structure from a set of `panels` that enables O(log N)
evaluation of the panel influence instead of O(N). For N ≥ 300, this is a huge speed-up.

**Note** the panel geometry, the tree, the potental `ϕ` and the strength `q` are all
bundled into the structure for efficient evaluation later.

# Fields
- `panels::Table`: Panel geometry with strength `q` (initialized to zero)
- `nodes::Table`: Aggregated node values in the BVH tree
- `bvh::BVH`: Bounding volume hierarchy of the panels and nodes

# Example
```julia
using NeumannKelvin
S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π, 0, 2π, hᵤ=1/16, N_max=3214) # quite a few panels
BH = BarnesHut(panels)
```

See also: [`BarnesHutSolve!`](@ref)
"""
struct BarnesHut{TP,TN,TB,F,KW}
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
function Base.show(io::IO, ::MIME"text/plain", BH::BarnesHut)
    show(io,BH);println()
    bbox = BH.bvh.nodes[1]
    println(io, "  bounds: $(bbox.lo) to $(bbox.up)")
    println(io, "  total area: ", BH.nodes[1].dA)
    println(io, "  strength extrema: $(extrema(BH.panels.q))")
end
# Compact version for collections
function Base.show(io::IO, BH::BarnesHut)
    print(io, "BarnesHut($(length(BH.panels)) panels, $(BH.bvh.tree.levels) levels)")
end

"""
    BarnesHutSolve(panels::Table,b;...) = BarnesHutSolve!(BarnesHut(panels;...),b;...)

See: [`BarnesHut`](@ref), [`BarnesHutSolve!`](@ref)
"""
BarnesHutSolve(panels,b=components(panels.n,1);ϕ=∫G,atol=1e-3,d²=4,kwargs...) = BarnesHutSolve!(BarnesHut(panels;ϕ,kwargs...),b;atol,d²)
using Krylov,LinearOperators
"""
    BarnesHutSolve!(BH::BarnesHut, b=components(BH.panels.n,1);
                    atol=1e-3, d²=4, verbose=true) -> BarnesHut

Solve the BEM linear system using Barnes-Hut acceleration and GMRES iteration.

Solves for panel strengths `q` such that the normal velocity boundary condition
is satisfied: `Σⱼ ∂ₙϕ(xᵢ,xⱼ)qⱼ = bᵢ`. The Barnes-Hut approximation uses distance
threshold `d²` to decide when to use aggregated node values vs individual panels.

# Arguments
- `BH::BarnesHut`: Pre-constructed Barnes-Hut solver (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `atol=1e-3`: Absolute tolerance for GMRES convergence
- `d²=4`: Barnes-Hut distance threshold (r²/R² > d² uses approximation)
- `verbose=true`: Print GMRES convergence statistics

# Returns
Modified `BH` with updated panel strengths in `BH.panels.q`

# Performance
- **Complexity**: O(N log N) per GMRES iteration vs O(N²) for direct
- **Typical**: 3-5 GMRES iterations for well-conditioned problems
- **Accuracy**: ~0.2% error vs direct solve with d²=4

# Example
```julia
BH = BarnesHut(panels)  # crazy fast initialization
BarnesHutSolve!(BH)     # fast solve
c_p = cₚ(BH)            # very fast measure
```

See also: [`BarnesHut`](@ref)
"""
function BarnesHutSolve!(BH,b=components(BH.panels.n,1);atol=1e-3,d²=4,verbose=true)
    # Make LinearOperator
    mult!(b,q) = (set_q!(BH,q); uₙ!(b,BH;d²))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b; atol)
    verbose && println(stats)
    set_q!(BH,q)
end
@inline function set_q!(BH,q)
    BH.panels.q .= q
    accumulate!(BH.nodes.q,BH.panels.dA .* q,BH.bvh)
    BH.nodes.q ./= BH.nodes.dA; BH
end
import AcceleratedKernels as AK
@inline uₙ!(b,(;panels,nodes,bvh,ϕ,kwargs);d²=4) = AK.foreachindex(b) do i
    b[i] = derivative(t->evaluate((x,p)->p.q*ϕ(x,p;kwargs...),panels.x[i]+t*panels.n[i],bvh,nodes,panels;d²),0.)
end

"""
    cₚ!(b,BH::BarnesHut;U=SVector(-1,0,0))
    cₚ(BH::BarnesHut;U=SVector(-1,0,0)) -> b

Measure the pressure coefficient on each panel induced by a solved panel tree
using Barnes-Hut and multi-threading to reduce cost from O(N²) to O(N log N / threads).

See also: [`BarnesHut`](@ref) [`BarnesHutSolve!`](@ref)
"""
cₚ!(b,BH;U=SVector(-1,0,0)) = AK.foreachindex(b) do i
    b[i] = 1-sum(abs2,U+∇Φ(BH.panels.x[i],BH))/sum(abs2,U)
end
cₚ(BH;U=SVector(-1,0,0)) = (b=similar(BH.panels.q);cₚ!(b,BH;U);b)
"""
    steady_force(BH::BarnesHut;U=SVector(-1,0,0))

Measure the integrated steady force induced by a solved panel tree
using Barnes-Hut and multi-threading to reduce cost from O(N²) to O(N log N / threads).

See also: [`BarnesHut`](@ref) [`BarnesHutSolve!`](@ref)
"""
function steady_force(BH;U=SVector(-1,0,0))
    panels = BH.panels; init=neutral=zero(eltype(panels.n))
    AK.mapreduce(+,panels,AK.get_backend(panels.q);init,neutral) do pᵢ
        cₚ = 1-sum(abs2,U+∇Φ(pᵢ.x,BH))/sum(abs2,U)
        cₚ*pᵢ.n*pᵢ.dA
    end
end
@inline Φ(x,(;panels,nodes,bvh,ϕ,kwargs)::BarnesHut) = evaluate((x,p)->p.q*ϕ(x,p,kwargs...),x,bvh,nodes,panels)

# Accumulate leaf values onto nodes
using ImplicitBVH
using ImplicitBVH: level_indices,pow2,unsafe_isvirtual
function accumulate!(node_values, leaf_values, bvh)
    tree = bvh.tree; levels = tree.levels
    # leaf level
    leaf = 0
    for i in range(level_indices(tree,levels-1)...)
        @inbounds node_values[i] = leaf_values[bvh.leaves[leaf+=1].index]
        leaf==length(leaf_values) && break
        @inbounds node_values[i] += leaf_values[bvh.leaves[leaf+=1].index]
    end

    # node levels
    for level in levels-2:-1:1
        parent,child = @inbounds bvh.skips[level:level+1]
        for i in pow2(level-1):pow2(level) - 1 - (child-parent)
            @inbounds node_values[i-parent] = node_values[2i-child]
            unsafe_isvirtual(tree,2i+1) && continue
            @inbounds node_values[i-parent] += node_values[2i+1-child]
        end
    end
    node_values
end
accumulate(leaf_values,bvh) = accumulate!(similar(leaf_values,length(bvh.nodes)), leaf_values, bvh)

# Relative squared-distance from bounding volumes
using ImplicitBVH: BBox, BSphere, BoundingVolume
reldist(x,bb::BSphere) = max(sum(abs2,x .- bb.x)/bb.r^2-1,0)
function reldist(x,bb::BBox)
    c = (bb.up .+ bb.lo) ./ 2
    r = (bb.up .- bb.lo) ./ 2
    q = abs.(x .- c) .- r
    sum(abs2,max.(q,0))/sum(abs2,r)
end
reldist(x,bb::BoundingVolume) = reldist(x,bb.volume)

# Barnes-Hut kernel evaluation
using ImplicitBVH: memory_index
function evaluate(fnc,x,bvh,node_values,leaf_values;d²=4,stack=Vector{Int}(undef,bvh.tree.levels))
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    top = 1; stack[top] = 1
    val = zero(fnc(x,leaf_values[1]))
    while top>0
        i = stack[top]; top-=1
        j = memory_index(tree,i)
        if j ≤ length_nodes
            if reldist(x,bvh.nodes[j])>d²
                val += fnc(x,node_values[j])
            else
                stack[top+=1] = 2i;
                !unsafe_isvirtual(tree,2i+1) && (stack[top+=1] = 2i+1)
            end
        else
            val += fnc(x,leaf_values[bvh.leaves[j-length_nodes].index])
        end
    end
    val
end

# panel bounding-box and node info
function bb_panel(panel)
    ext = extrema.(components(panel.xᵤᵥ))
    BBox(first.(ext),last.(ext))
end
bvh_panels(panels) = BVH(bb_panel.(panels))
function fill_nodes(panels,bvh)
    dA = accumulate(panels.dA,bvh)
    x = accumulate(panels.dA .* panels.x, bvh) ./ dA
    n = NeumannKelvin.normalize.(accumulate(panels.dA .* panels.n, bvh))
    Table(;x,dA,n)
end