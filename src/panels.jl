"""
    panelize(surface,args...;kwargs...) = measurepanel.(surfsplit(surface,args...;kwargs...))

See `measurepanel` & `surfsplit`
"""
panelize(surface,args...;kwargs...) = map(surfsplit(surface,args...;kwargs...)) do uv
    x,d = (uv.low+uv.high)/2,(uv.low-uv.high)
    measurepanel(surface,x...,d...;kwargs...)
end
"""
    surfsplit(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,depthmax=10,kwargs...)

Recurvsively split a `surface` defined over `u∈[u₀,u₁]` and `v∈[v₀,v₁]`, until the panels
are roughly `hᵤ,hᵥ` in size, or the function hits the recursion `depthmax`. The parameter
`(hᵤ+hᵥ)*c` sets the max deviation of the panel from a flat plane by reducing panel size
in regions of high curvature.
"""
function surfsplit(surface,u₀=0.,u₁=1.,v₀=0.,v₁=1.;hᵤ=1.,hᵥ=hᵤ,c=0.05,depthmax=14,kwargs...)
    # Check inputs and get output type
    (u₀≥u₁ || v₀≥v₁) && throw(ArgumentError("Need `u₀<u₁` and `v₀<v₁`. Got [$u₀,$u₁],[$v₀,$v₁]."))
    (hᵤ≤0 || hᵥ≤0 || c≤0) && throw(ArgumentError("Need positive `hᵤ,hᵥ,c`. Got $hᵤ,$hᵥ,$c."))
    !(typeof(surface(u₀,v₀)) <: SVector) && throw(ArgumentError("`surface` function doesn't return an SVector."))
    depthmax>14 && @warn "Number of panels can be up to 2^depthmax!"

    # Initialize the arc data
    u = arcsplit(u->surface(u,0.5v₀+0.5v₁),u₀,u₁,hᵤ,c)
    v = arcsplit(v->surface(0.5u₀+0.5u₁,v),v₀,v₁,hᵥ,c)
    (u.len < 0.5hᵤ || v.len < 0.5hᵥ) && return [] # not big enough!

    # Recursively split the surface
    _surfsplit(surface,u,v,hᵤ,hᵥ,c,0,depthmax) |> Table
end

"""
    arcsplit(r, u₀, u₁, Δs, c) -> (len=S,low=u₀,mid=û,high=u₁)

Find the pseudo-arclength `S = ∫s'(u) du` and centroid `û = S⁻¹∫u s'(u) du`
along a curve `r(u), u ∈ [u₀, u₁]` by integrating a pseudo-arcspeed `s'`.
The inputs `Δs, c` are a segment length and deviation, as explained below.

# Details

The speed `s'` is defined as

    s' = max(l',√(Δs*κₙ/8c)))

where `l' ≡ ∥r'∥` and `κₙ=√(∥r"∥²-l"²)` are the arcspeed and normal curvature,
such that segments of length `Δs` along `r` deviate no more than `δ≤Δs*c` from
the curve. Starting from the curvature-based deviation estimate `δ≈Δl²/8κ`,
and using `Δl≈l'*Δu` and `Δs≈s'*Δu` gives the inequality `s'≥l'√(Δs*κ/8c)`.
Demanding also that `s'≥l'` such that `Δl≤Δs` and substituting `κₙ = l'² κ`
leads to the equation for `s'` above.
"""
function arcsplit(r,u₀,u₁,Δs,c,x=xgl4,w=wgl4)
    @inline speed(u) = max(arcspeed(r,u),√(Δs*κₙ(r,u)/8c))
    h,cen = (u₁-u₀)/2,(u₁+u₀)/2 # interval scales
    u = h*x .+ cen    # points
    s = speed.(u)     # values
    S = h*w's         # arclength
    û = h*w'*(u.*s)/S # centroid
    (len=S,low=u₀,mid=û,high=u₁)
end
arcspeed(r,u) = norm(derivative(r,u))
κₙ(r,u) = √max(0,sum(abs2,derivative(u->derivative(r,u),u))-derivative(u->arcspeed(r,u),u)^2)

_surfsplit(surf,u,v,hᵤ,hᵥ,c,depth,depthmax) = if depth≥depthmax && return combine(u,v)
elseif u.len/hᵤ ≥ v.len/hᵥ
    u.len ≤ hᵤ && return combine(u,v)
    mapreduce(vcat,((u.low,u.mid),(u.mid,u.high))) do (u₀,u₁)
        uₛ = arcsplit(u->surf(u,v.mid),u₀,u₁,hᵤ,c)         # split
        vₛ = arcsplit(v->surf(uₛ.mid,v),v.low,v.high,hᵥ,c) # remeasure
        _surfsplit(surf,uₛ,vₛ,hᵤ,hᵥ,c,depth+1,depthmax)
    end
else
    v.len ≤ hᵥ && return combine(u,v)
    mapreduce(vcat,((v.low,v.mid),(v.mid,v.high))) do (v₀,v₁)
        vₛ = arcsplit(v->surf(u.mid,v),v₀,v₁,hᵥ,c)         # split
        uₛ = arcsplit(u->surf(u,vₛ.mid),u.low,u.high,hᵤ,c) # remeasure
        _surfsplit(surf,uₛ,vₛ,hᵤ,hᵥ,c,depth+1,depthmax)
    end
end
combine(a::NamedTuple{K}, b::NamedTuple{K}) where K = NamedTuple{K}(map(k -> SA[a[k],b[k]], K))

const Δg,Δx = SA[-0.5/√3,0.5/√3],SA[-0.5,0.5]
using HCubature
"""
    measurepanel(S,u,v,du,dv;flip=false,cubature=false) -> (x,n,dA,x₄,w₄)

Measures a parametric surface function `S(u,v)` for a `u,v ∈ [u±0.5du]×[v±0.5dv]` panel.
Returns centroid point and normal `x,n`, the surface area `dA`, and the 2x2 Gauss-point
locations and weights `x₄,w₄`. Panel corner data `xᵤᵥ,nᵤᵥ` is used only for plotting.
 - `flip=true` flips the panel to point the other way.
 - `cubature=true` uses an adaptive "h-cubature" for `dA,x,n`.
"""
function measurepanel(S,u,v,du,dv;flip=false,cubature=false,kwargs...)
    flip && return measurepanel((v,u)->S(u,v),v,u,dv,du;cubature)
    # get 2x2 Gauss-points
    x₄ = S.(u .+ du*Δg, v .+ dv*Δg')
    n₄ = normal.(S, u .+ du*Δg, v .+ dv*Δg')
    dA₄ = norm.(n₄)*du*dv/4 # area-scaled weights (normalized at end)
    # get area
    cube(f) = hcubature(f,SA[u-0.5du,v-0.5dv],SA[u+0.5du,v+0.5dv],rtol=0.01)[1]
    dA = cubature ? cube(uv->norm(normal(S,uv...))) : sum(dA₄)
    # get centroid
    x = cubature ? cube(uv->S(uv...)*norm(normal(S,uv...)))/dA : sum(x₄ .* dA₄)/dA
    n = cubature ? normalize(cube(uv->normal(S,uv...))) : normalize(sum(n₄))
    # get corners (only for pretty plots)
    xᵤᵥ = S.(u .+ du*Δx, v .+ dv*Δx')
    nᵤᵥ = normalize.(normal.(S, u .+ du*Δx, v .+ dv*Δx'))
    # combine everything into named tuple
    (x=x, n=n, dA=dA, x₄=x₄, w₄=dA₄ .* dA/sum(dA₄), xᵤᵥ=xᵤᵥ, nᵤᵥ=nᵤᵥ)
end
normal(S,u,v) = derivative(u->S(u,v),u)×derivative(v->S(u,v),v)
normalize(v::SVector{n,T}) where {n,T} = v/(eps(T)+norm(v))
"""
    deviation = distance from panel center to plane defined by the corners
"""
function deviation(p)
    a =   0.5p.xᵤᵥ[1,1]+0.5p.xᵤᵥ[1,2] # plane base
    l = a-0.5p.xᵤᵥ[2,1]-0.5p.xᵤᵥ[2,2] # vector to plane top
    hypot((p.x-a-l*(p.x-a)'l/l'l...)) # distance from center
end