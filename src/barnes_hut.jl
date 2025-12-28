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
cen = SA[0,0,1]
S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]+cen
# panels = measure_panel.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2,cubature=true) |> Table
panels = panelize(S,0,π,0,2π,hᵤ=0.12)
bvh = bvh_panels(panels)
nodes = fill_nodes(panels,bvh)
@assert nodes.dA[1]≈sum(panels.dA)
@assert nodes.x[1]≈sum(panels.x .* panels.dA)/sum(panels.dA)≈cen

using ForwardDiff
ρ = panels.x[length(panels)÷3]-cen
for r in 1:6
    x = r*ρ+cen
    error = evaluate(∫G,x,bvh,nodes,panels)/sum(∫G(x,p,d²=Inf) for p in panels)-1
    derror = ForwardDiff.gradient(x->evaluate(∫G,x,bvh,nodes,panels),x) ./ ForwardDiff.gradient(x->sum(∫G(x,p,d²=Inf) for p in panels),x) .- 1
    @show r,error,derror
end

using BenchmarkTools
stack = Vector{Int}(undef,bvh.tree.levels)
x = panels.x[1]
d = ForwardDiff.Dual.(panels.x[1],panels.n[1])
p = panels[1]
@btime evaluate(∫G,$x,$bvh,$nodes,$panels,stack=$stack)
@btime evaluate(∫G,$d,$bvh,$nodes,$panels,stack=$stack)

for h in 0.5 .^ (1:6)
    panels = panelize(S,0,π,0,2π,hᵤ=h,N_max=Inf)
    bvh = bvh_panels(panels)
    nodes = fill_nodes(panels,bvh)
    @show h,length(panels)
    for r in (1,6)
        @show r
        @btime sum(panel->∫G($r*ρ+cen,panel,d²=Inf),panels)
        @btime bhut = evaluate(∫G,$r*ρ+cen,bvh,nodes,panels)
    end
end