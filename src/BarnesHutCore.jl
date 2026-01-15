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
        parent,child = @inbounds bvh.skips[level],bvh.skips[level+1]
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
function evaluate(fnc, x, bvh, node_values, leaf_values; d²=4,
                  val=zero(fnc(x,leaf_values[1])), verbose=false)
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    node_count = leaf_count = 0
    # Depth-First-Search
    i = 1; while true
        @inbounds j = memory_index(tree,i)
        if j ≤ length_nodes
            # check if we can prune at this node
            if reldist(x,@inbounds bvh.nodes[j])>d²
                val += fnc(x,@inbounds node_values[j])
                verbose && (node_count+=1)
            else # otherwise decend to child and continue
                i = 2i; continue
            end
        else # use the leaf value
            @inbounds j = bvh.leaves[j-length_nodes].index
            val += fnc(x,@inbounds leaf_values[j])
            verbose && (leaf_count+=1)
        end
        # get next sibling index
        i = i>>trailing_ones(i)+1
        (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
    end
    verbose && println("evaluated: node count=$node_count, leaf count=$leaf_count")
    val
end

# panel bounding-box
ImplicitBVH.BoundingVolume(panel::NamedTuple) = if hasproperty(panel,:verts)
    ext = extrema.(components(panel.verts))
    return BBox(first.(ext),last.(ext))
else
    return BSphere(panel.x,√panel.dA)
end