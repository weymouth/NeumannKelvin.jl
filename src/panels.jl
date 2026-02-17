"""
    panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,devlimit=5f-2,
            transpose=false,flip=false,submerge=false,N_max=1000,kwargs...)

Panelize a parametric `surface` of `u∈[u₀,u₁]` and `v∈[v₀,v₁]`, returning a `Table` of panels.

The surface is split into strips roughly `hᵤ` wide which are split into panels roughly `hᵥ` high which are then `measure`d.
The parameter `devlimit` sets the max deviation `δ/(hᵤ+hᵥ)` from the surface by reducing panel size in regions of high curvature.
Use `transpose=true` to change the strip direction, `flip=true` to flip the normal direction, and submerge=`true` to trim surface for `z≥0`.
This function throws an error if the adaptive routine gives more than `N_max` panels.
"""
function panelize(surface,u₀=0,u₁=1f0,v₀=0,v₁=1f0;hᵤ=1,hᵥ=hᵤ,devlimit=5f-2,
                  transpose=false,flip=false,N_max=1000,verbose=false,submerge=false,kwargs...)
    # Transpose arguments u,v -> v,u
    transpose && return panelize((v,u)->surface(u,v),v₀,v₁,u₀,u₁,;hᵤ=hᵥ,hᵥ=hᵤ,devlimit,
                                 transpose=false,flip=!flip,submerge,N_max,verbose,kwargs...)

    # Check inputs and get output type
    (u₀≥u₁ || v₀≥v₁) && throw(ArgumentError("Need `u₀<u₁` and `v₀<v₁`. Got [$u₀,$u₁],[$v₀,$v₁]."))
    (hᵤ≤0 || hᵥ≤0 || devlimit≤0) && throw(ArgumentError("Need positive `hᵤ,hᵥ,devlimit`. Got $hᵤ,$hᵥ,$devlimit."))
    u₀,u₁,v₀,v₁ = promote(u₀,u₁,v₀,v₁)
    verbose && @show u₀,u₁,v₀,v₁
    !(typeof(surface(u₀,v₀)) <: SVector) && throw(ArgumentError("`surface` function doesn't return an SVector."))
    init = typeof(measure(surface,(u₀+u₁)/2,(v₀+v₁)/2,u₁-u₀,v₁-v₀))[]

    # Get arclength and inverse function along bottom & top edges
    S₀,s₀⁻¹ = arclength(u->surface(u,v₀),hᵤ,devlimit,u₀,u₁)
    S₁,s₁⁻¹ = arclength(u->surface(u,v₁),hᵤ,devlimit,u₀,u₁)
    verbose && @show S₀,S₁

    # Define strips
    min(S₀,S₁) ≤ hᵤ/2 && return Table(init)      # check min width
    S,_,s⁻¹ = max((S₀,1,s₀⁻¹),(S₁,2,s₁⁻¹))        # get longer edge
    N = round(Int,S/hᵤ); uᵢ = s⁻¹(range(0,S,N+1)) # define strips
    verbose && @show N

    # Mapreduce across strips
    panels = mapreduce(vcat,1:N; init) do i
        u,du = (uᵢ[i+1]+uᵢ[i])/2,uᵢ[i+1]-uᵢ[i] # Parametric center & width

        # Find equidistant points along strip center
        uv₀,uv₁ = submerge ? newlimits(v->surface(u,v),v₀,v₁) : (v₀,v₁)
        S,s⁻¹ = arclength(v->surface(u,v),hᵥ,devlimit,uv₀,uv₁)
        verbose && @show i,S
        S ≤ hᵥ/2 && return init               # not enough height
        ve = s⁻¹(range(0,S,round(Int,S/hᵥ)+1)) # panel endpoints
        verbose && @show length(ve)-1
        v = (ve[2:end]+ve[1:end-1])/2          # panel centers
        dv = ve[2:end]-ve[1:end-1]             # panel heights

        # Measure panels along strip
        @. measure(surface,u,v,du,dv;flip,kwargs...)
    end

    # Check length and return as a Table
    length(panels) ≤ N_max && return Table(panels)
    throw(ArgumentError("length(panels)=$(length(panels))>$N_max. Increase hᵤ,hᵥ,devlimit and/or N_max."))
end
function newlimits(r,low,high)
    z(u) = r(u)[3]+√eps(u)
    zero = find_zero(z,(low,high))
    return z(low)<0 ? (low,zero) : (zero,high)
end
using QuadGK,DataInterpolations
"""
    arclength(r, Δs, devlimit, low, high) -> S, u(s)

Find the pseudo-arclength `s` along a curve `r(u), u ∈ [low,high]`
by integrating the pseudo-arcspeed `s' = max(l',√(Δs*aₙ/8devlimit)))` where
`l' ≡ ∥r'∥` and `aₙ=√(∥r"∥²-l"²)` are the arcspeed and normal acceleration.

# Returns

- `S`: the total pseudo-arclength over `[low,high]`,
- `u(s)`: a monotonic spline for the inverse mapping.

# Details

The speed `s'` is defined such that segments of length `Δs` along `r`
deviate no more than `δ≤Δs*devlimit` from the curve. Starting from the
curvature-based deviation estimate `δ≈Δl²κ/8`, and using `Δl≈l'*Δu`
and `Δs≈s'*Δu` gives the inequality `s'≥l'√(Δs*κ/8devlimit)`. Demanding also
that `s'≥l'` such that `Δl≤Δs` and substituting `aₙ = l'² κ` leads to
the rate equation above.
"""
function arclength(r,Δs,devlimit,low,high)
    # Use quadgk to adaptively sample ∫ds, returning S and subintervals Δᵢ
    @inline speed(u) = max(arcspeed(r)(u),√(Δs*aₙ(r,u)/8devlimit))
    S,_,Δᵢ = quadgk_segbuf(speed,range(low,high,4),rtol=1e-5,order=3)
    # Order the subintervals and accumulate the arclength data s(uᵢ)=∑ⁱⱼ₌₀ Δsⱼ
    sort!(Δᵢ,by=Δ->Δ.a)
    uᵢ,sᵢ = [low; map(Δ->Δ.b,Δᵢ)],[zero(S); cumsum(map(Δ->Δ.I,Δᵢ))]
    # Construct a Monotonic Hermite Cubic Spline for u(s) using bounded duᵢ/ds
    duᵢ = map(eachindex(uᵢ)) do i
        min(3secant(Δᵢ[max(1,i-1)]),3secant(Δᵢ[min(i,length(Δᵢ))]),inv(speed(uᵢ[i])))
    end
    S,CubicHermiteSpline(duᵢ,uᵢ,sᵢ,extrapolation = ExtrapolationType.Constant)
end
arcspeed(r) = u->norm(derivative(r,u))
aₙ(r,u) = √max(0,sum(abs2,derivative(u->derivative(r,u),u))-derivative(arcspeed(r),u)^2)
secant(Δ)=(Δ.b-Δ.a)/Δ.I

abstract type GreenKernel end
struct QuadKernel <: GreenKernel end
using HCubature
"""
    measure(S,u,v,du,dv;flip=false,cubature=false) -> (x,n,dA,xg,wg)

Measures a parametric surface function `S(u,v)` for a `u,v ∈ [u±du/2]×[v±dv/2]` panel.
Returns centroid point and normal `x,n`, the surface area `dA`, and the Gauss-point
locations and weights `xg,wg`. Panel corner data `vertices,nvertices` is used only for plotting.
 - `flip=true` flips the panel to point the other way.
 - `cubature=true` uses an adaptive "h-cubature" for `dA,x,n`.
"""
function measure(S,u,v,du,dv;flip=false,cubature=false,Δg=SA_F32[-1/√3,1/√3],wg=SA[1,1])
    flip && return measure((v,u)->S(u,v),v,u,dv,du;cubature,Δg,wg)
    # get Gauss-points
    x₄ = S.(u .+ du*Δg/2, v .+ dv*Δg'/2)
    n₄ = normal.(S, u .+ du*Δg/2, v .+ dv*Δg'/2)
    dA₄ = norm.(n₄).*(wg*wg')*du*dv/4 # area-scaled weights
    # get area
    cube(f) = hcubature(f,SA[u-du/2,v-dv/2],SA[u+du/2,v+dv/2],rtol=0.01)[1]
    dA = cubature ? cube(uv->norm(normal(S,uv...))) : sum(dA₄)
    # get centroid
    x = cubature ? cube(uv->S(uv...)*norm(normal(S,uv...)))/dA : sum(x₄ .* dA₄)/dA
    n = cubature ? normalize(cube(uv->normal(S,uv...))) : normalize(sum(n₄.*(wg*wg')))
    # get corners (only for pretty plots)
    xᵤᵥ = S.(u .+ SA[-du,du]/2, v .+ SA[-dv,dv]'/2)
    nᵤᵥ = normalize.(normal.(S, u .+ SA[-du,du]/2, v .+ SA[-dv,dv]'/2))
    # combine everything into named tuple
    (x=x, n=n, dA=dA, xg=x₄, wg=dA₄ .* dA/sum(dA₄), verts=unwrap(xᵤᵥ), nverts=unwrap(nᵤᵥ), kernel=QuadKernel())
end
normal(S,u,v) = derivative(u->S(u,v),u)×derivative(v->S(u,v),v)
normalize(v::SVector{n,T}) where {n,T} = v/(eps(T)+norm(v))
unwrap(a) = map(i->a[i],SA[1,2,4,3])