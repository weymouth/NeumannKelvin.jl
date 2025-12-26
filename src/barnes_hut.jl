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
    level_offset = cumsum(bvh.skips)
    for level in levels-2:-1:1
        parent = level_offset[level]; child = level_offset[level+1]
        for i in pow2(level-1):pow2(level) - 1 - @inbounds bvh.skips[level]
            @inbounds node_values[i-parent] = node_values[2i-child]
            unsafe_isvirtual(tree,2i+1) && continue
            @inbounds node_values[i-parent] += node_values[2i+1-child]
        end
    end
    node_values
end
accumulate(leaf_values::T,bvh) where T = accumulate!(T(undef,length(bvh.nodes)), leaf_values, bvh)

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

# BEM-specific stuff!!
using NeumannKelvin
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

# Test it
S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
panels = measure_panel.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2,cubature=true) |> Table
bvh = bvh_panels(panels)
nodes = fill_nodes(panels,bvh)

using ForwardDiff
θ = π/5; ρ = SA[cos(θ),sin(θ)*cos(θ),sin(θ)*sin(θ)]
map(1:6) do r
    dx = ForwardDiff.Dual.(r*ρ,ones(typeof(ρ)))
    r,evaluate(∫G,dx,bvh,nodes,panels)/sum(∫G(dx,p,d²=0) for p in panels)-1
end