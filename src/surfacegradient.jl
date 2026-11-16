"""
    loopweights(panels, cuts, mirrors)

Weights `wᵢⱼ` such that `uᵢ = -4π∑ⱼwᵢⱼqⱼ` is the surface velocity on dipole panel `i`: the mean of ∇Φ over the
region enclosed by a loop through the centroids of the panels around `i`, using ∫∇Φ da = ∮Φ m dl with
Φ=-4πqⱼ on panel `j`. The loop links the centroids around each point on the edge of panel `i`, across the
edges leaving that point, but never across the trailing edges `cuts`. Mirror images close the loops on
symmetry planes.
"""
function loopweights(panels,cuts,mirrors)
    P,C,owner = images(panels,cuts,mirrors)
    scale = maximum(p->maximum(norm,p.verts),panels); key(v) = round.(v/scale,digits=8) .+ 0.0
    at = Dict{eltype(P.x),Vector{Int}}()
    for (j,p) in enumerate(P), v in p.verts; push!(get!(at,key(v),Int[]),j); end
    w = Vector{Vector{Pair{Int,eltype(panels.x)}}}(undef,length(panels))
    AK.foreachindex(w) do i
        J = setdiff(reduce(vcat,(at[key(v)] for v in P.verts[i])),i)
        w[i] = [owner[j]=>wⱼ for (j,wⱼ) in loopweights(i,P,J,C)]
    end; w
end
function images(panels,cuts,mirrors) # panels and cuts plus the mirror images touching a symmetry plane
    P,C,owner = collect(panels),collect(cuts),collect(eachindex(panels))
    for m in mirrors, (j,p) in enumerate(panels)
        all(isone,m) && continue
        any(v->v .* m ≈ v,p.verts) || continue
        push!(P,merge(measure(reverse(map(v->v .* m,p.verts))...),(;p.kernel,p.q))); push!(owner,j)
    end
    for m in mirrors; all(isone,m) || append!(C,map(c->map(v->v .* m,c),cuts)); end
    Table(P),C,owner
end
function loopweights(i,panels,J,cuts)
    link(a,b,v) = (e = shared(panels[a],panels[b]); !isnothing(e) && !any(c->along(e,c),cuts) && along(v,e))
    segs = Tuple{Int,eltype(panels.x),eltype(panels.x)}[]
    for v in nodes(i,panels,J)
        group,around = [i],filter(j->onedges(v,panels[j]),J) # panels reachable around v from i without crossing a cut
        while (k = findfirst(b->b∉group && any(a->link(a,b,v),group),around))≠nothing; push!(group,around[k]); end
        for a in group[2:end]
            any(c->along(v,c) && onedges(c,panels[a]),cuts) && push!(segs,(a,panels.x[a],v))
            for b in group[2:end]; a<b && link(a,b,v) && push!(segs,crossing(a,b,panels,panels.n[i])...); end
        end
    end
    foreach(c->onedges(c,panels[i]) && push!(segs,(i,c...)),cuts)
    x,n = panels.x[i],panels.n[i]; loop = chain(segs)
    area = sum(((j,f,t),)->(f-x)×(t-x),loop)'n/2 # signed, so the weights don't depend on the loop's direction
    [j=>(t-f)×n/area for (j,f,t) in loop]
end
function crossing(a,b,panels,n) # centroid a to centroid b, split where the line crosses their shared edge
    xa,xb,(e₁,e₂) = panels.x[a],panels.x[b],shared(panels[a],panels[b])
    d,e = xb-xa,e₂-e₁; X = xa+((e₁-xa)×e)'n/((d×e)'n)*d
    (a,xa,X),(b,X,xb)
end
function chain(segs) # join the segments end-to-end into a closed loop
    loop = [popfirst!(segs)]
    while !isempty(segs)
        k = findfirst(s->s[2]≈last(loop)[3] || s[3]≈last(loop)[3],segs)
        isnothing(k) && throw(ArgumentError("open loop: the lifting body must be closed"))
        j,f,t = popat!(segs,k); push!(loop, f≈last(loop)[3] ? (j,f,t) : (j,t,f))
    end
    last(loop)[3]≈first(loop)[2] || throw(ArgumentError("open loop: the lifting body must be closed"))
    loop
end
nodes(i,panels,J) = foldl((u,v)->any(≈(v),u) ? u : push!(u,v),
    filter(v->onedges(v,panels[i]),[v for j in [i;J] for v in panels.verts[j]]);init=eltype(panels.x)[])
edges(p) = ((p.verts[k],p.verts[k%length(p.verts)+1]) for k in eachindex(p.verts))
along(v::SVector,(a,b)) = (d=b-a; t=(v-a)'d/(d'd); ϵ=√eps(typeof(t)); -ϵ<t<1+ϵ && norm(v-a-t*d)<ϵ*norm(d)) # v on segment ab
along((a,b)::Tuple,s) = along(a,s) && along(b,s) # segment ab within s
onedges(s,p) = any(e->along(s,e),edges(p)) # point or segment s on the edges of panel p
function shared(p,q) # the stretch of edge common to panels p and q, or nothing
    for e in edges(p), f in edges(q)
        (e[1]≈e[2] || f[1]≈f[2]) && continue
        along(f,e) && return f
        along(e,f) && return e
    end
end