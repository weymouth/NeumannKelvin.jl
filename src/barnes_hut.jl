using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}([1., 0., 0.], 1.),
    BSphere{Float32}([2., 0., 0.], 1.),
    BSphere{Float32}([3., 0., 0.], 1.),
    BSphere{Float32}([1., 1., 1.], 1.),
    BSphere{Float32}([2., 1., 1.], 1.),
    BSphere{Float32}([3., 1., 1.], 1.),
    BSphere{Float32}([1., 1., 2.], 1.),
    BSphere{Float32}([2., 1., 2.], 1.),
    BSphere{Float32}([3., 1., 2.], 1.),
]

# Build BVH
bvh = BVH(bounding_spheres)

# Accumulate leaf values onto node values
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

# Test it
leafm = Float32.(1:length(bounding_spheres))
nodem = zeros(Float32,length(bvh.nodes))
accumulate!(nodem,leafm,bvh)
using StaticArrays
leafx = [SA[bb.x...] for bb in bounding_spheres]
leafmx = leafm .* leafx
nodemx = zeros(SVector{3,Float32},length(bvh.nodes))
accumulate!(nodemx,leafmx,bvh)
nodex = nodemx ./ nodem

# Relative squared-distance from bounding volumes
reldist(x,bb::BSphere) = max(sum(abs2,x .- bb.x)/bb.r^2-1,0)
function reldist(x,bb::BBox)
    c = (bb.up .+ bb.lo) ./ 2
    r = (bb.up .- bb.lo) ./ 2
    q = abs.(x .- c) .- r
    sum(abs2,max.(q,0))/sum(abs2,r)
end
reldist(x,bb::ImplicitBVH.BoundingVolume) = reldist(x,bb.volume)

using ImplicitBVH: memory_index
function evaluate(fnc,x,bvh,node_values,leaf_values;d²=1,stack=Vector{Int}(undef,bvh.tree.levels))
    tree = bvh.tree; real_internal_nodes = tree.real_nodes-tree.real_leaves
    top = 1; stack[top] = 1
    val = zero(fnc(x,node_values[1]))
    while top>0
        i = stack[top]; top-=1
        j = memory_index(tree,i)
        if j ≤ real_internal_nodes
            if reldist(x,bvh.nodes[j])>d²
                val += fnc(x,node_values[j])
            else
                stack[top+=1] = 2i;
                !unsafe_isvirtual(tree,2i+1) && (stack[top+=1] = 2i+1)
            end
        else
            val += fnc(x,leaf_values[bvh.leaves[j-real_internal_nodes].index])
        end
    end
    val
end

using TypedTables
nodeT = Table(m=nodem,x=nodex)
leafT = Table(m=leafm,x=leafx)
fnc(x,data) = data.m/sum(abs2,x-data.x)

x = zero(SVector{3,Float32})
evaluate(fnc,x,bvh,nodeT,leafT)

using ForwardDiff
dx = ForwardDiff.Dual.(x,ones(SVector{3,Float32}))
evaluate(fnc,dx,bvh,nodeT,leafT)
