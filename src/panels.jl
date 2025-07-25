"""
    panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,flip=false,N_max=1000,kwargs...)

Panelize a parametric `surface` of `u∈[u₀,u₁]` and `v∈[v₀,v₁]`, returning a `Table` of panels.

The surface is split into strips roughly `hᵤ` wide, which are split into panels roughly `hᵥ` high.
Use `transpose=true` to change the strip direction and `flip=true` to flip the normal direction.
The parameter `(hᵤ+hᵥ)*c` sets the max deviation of the panel from a flat plane by reducing
panel size in regions of high curvature. Errors if the adaptive routine gives more than `N_max` panels.
"""
function panelize(surface,u₀=0.,u₁=1.,v₀=0.,v₁=1.;hᵤ=1.,hᵥ=hᵤ,c=0.05,
                  transpose=false,flip=false,N_max=1000,verbose=false,kwargs...)
    # Transpose arguments u,v -> v,u
    transpose && return panelize((v,u)->surface(u,v),v₀,v₁,u₀,u₁,;hᵤ=hᵥ,hᵥ=hᵤ,c,
                                 transpose=false,flip=!flip,N_max,verbose,kwargs...)

    # Check inputs and get output type
    (u₀≥u₁ || v₀≥v₁) && throw(ArgumentError("Need `u₀<u₁` and `v₀<v₁`. Got [$u₀,$u₁],[$v₀,$v₁]."))
    (hᵤ≤0 || hᵥ≤0 || c≤0) && throw(ArgumentError("Need positive `hᵤ,hᵥ,c`. Got $hᵤ,$hᵥ,$c."))
    !(typeof(surface(u₀,v₀)) <: SVector) && throw(ArgumentError("`surface` function doesn't return an SVector."))
    init = typeof(measure_panel(surface,0.5u₀+0.5u₁,0.5v₀+0.5v₁,u₁-u₀,v₁-v₀))[]

    # Get arcslength and inverse along bottom & top edges
    S₀,s₀⁻¹ = arclength(u->surface(u,v₀),hᵤ,c,u₀,u₁)
    S₁,s₁⁻¹ = arclength(u->surface(u,v₁),hᵤ,c,u₀,u₁)
    verbose && @show S₀,S₁

    # Define strips
    min(S₀,S₁) ≤ 0.5hᵤ && return Table(init)      # check min width
    S,_,s⁻¹ = max((S₀,1,s₀⁻¹),(S₁,2,s₁⁻¹))        # get longer edge
    N = round(Int,S/hᵤ); uᵢ = s⁻¹(range(0,S,N+1)) # define strips
    verbose && @show N

    # Mapreduce across strips
    panels = mapreduce(vcat,1:N; init) do i
        u,du = 0.5uᵢ[i+1]+0.5uᵢ[i],uᵢ[i+1]-uᵢ[i] # Parametric center & width

        # Find equidistant points along strip
        S,s⁻¹ = arclength(v->surface(u,v),hᵥ,c,v₀,v₁) # speed along strip center
        verbose && @show i,S
        S ≤ 0.5hᵥ && return init               # not enough height
        ve = s⁻¹(range(0,S,round(Int,S/hᵥ)+1)) # panel endpoints
        verbose && @show length(ve)
        v = 0.5*(ve[2:end]+ve[1:end-1])        # panel centers
        dv = ve[2:end]-ve[1:end-1]             # panel heights

        # Measure panels along strip
        @. measure_panel(surface,u,v,du,dv;flip,kwargs...)
    end

    # Check length and return as a Table
    length(panels) ≤ N_max && return Table(panels)
    throw(ArgumentError("length(panels)=$(length(panels))>$N_max. Increase hᵤ,hᵥ,c and/or N_max."))
end

using QuadGK,DataInterpolations
"""
    arclength(r, Δs, c, low, high) -> S, u(s)

Find the pseudo-arclength `s` along a curve `r(u), u ∈ [low,high]`
by integrating the pseudo-arcspeed `s' = max(l',√(Δs*κₙ/8c)))` where
`l' ≡ ∥r'∥` and `κₙ=√(∥r"∥²-l"²)` are the arcspeed and normal curvature.

# Returns

- `S`: the total pseudo-arclength over `[low,high]`,
- `u(s)`: a monotonic spline for the inverse mapping.

# Details

The speed `s'` is defined such that segments of length `Δs` along `r`
deviate no more than `δ≤Δs*c` from the curve. Starting from the
curvature-based deviation estimate `δ≈Δl²/8κ`, and using `Δl≈l'*Δu`
and `Δs≈s'*Δu` gives the inequality `s'≥l'√(Δs*κ/8c)`. Demanding also
that `s'≥l'` such that `Δl≤Δs` and substituting `κₙ = l'² κ` leads to
the rate equation above.
"""
function arclength(r,Δs,c,low,high)
    # Use quadgk to adaptively sample ∫ds, returning S and subintervals Δᵢ
    @inline speed(u) = max(arcspeed(r)(u),√(Δs*κₙ(r,u)/8c))
    S,_,Δᵢ = quadgk_segbuf(speed,range(low,high,4),rtol=1e-5,order=3)
    # Order the subintervals and accumulate the arclength data s(uᵢ)=∑ⁱⱼ₌₀ Δsⱼ
    sort!(Δᵢ,by=Δ->Δ.a)
    uᵢ,sᵢ = [low; map(Δ->Δ.b,Δᵢ)],[zero(S); cumsum(map(Δ->Δ.I,Δᵢ))]
    # Construct a Monotonic Hermite Cubic Spline for u(s) using bounded duᵢ/ds
    duᵢ = map(eachindex(uᵢ)) do i
        min(3secant(Δᵢ[max(1,i-1)]),3secant(Δᵢ[min(i,length(Δᵢ))]),inv(speed(i==1 ? Δᵢ[1].a : Δᵢ[i-1].b)))
    end
    S,CubicHermiteSpline(duᵢ,uᵢ,sᵢ,extrapolation = ExtrapolationType.Constant)
end
arcspeed(r) = u->norm(derivative(r,u))
κₙ(r,u) = √max(0,sum(abs2,derivative(u->derivative(r,u),u))-derivative(arcspeed(r),u)^2)
secant(Δ)=(Δ.b-Δ.a)/Δ.I

using HCubature
"""
    measure_panel(S,u,v,du,dv;flip=false,cubature=false) -> (x,n,dA,x₄,w₄)

Measures a parametric surface function `S(u,v)` for a `u,v ∈ [u±0.5du]×[v±0.5dv]` panel.
Returns centroid point and normal `x,n`, the surface area `dA`, and the 2x2 Gauss-point
locations and weights `x₄,w₄`. Panel corner data `xᵤᵥ,nᵤᵥ` is used only for plotting.
 - `flip=true` flips the panel to point the other way.
 - `cubature=true` uses an adaptive "h-cubature" for `dA,x,n`.
"""
function measure_panel(S,u,v,du,dv;flip=false,cubature=false,Δg=SA[-1/√3,1/√3],wg=SA[1,1])
    flip && return measure_panel((v,u)->S(u,v),v,u,dv,du;cubature)
    # get Gauss-points
    x₄ = S.(u .+ 0.5du*Δg, v .+ 0.5dv*Δg')
    n₄ = normal.(S, u .+ 0.5du*Δg, v .+ 0.5dv*Δg')
    dA₄ = norm.(n₄).*(wg*wg')*du*dv/4 # area-scaled weights
    # get area
    cube(f) = hcubature(f,SA[u-0.5du,v-0.5dv],SA[u+0.5du,v+0.5dv],rtol=0.01)[1]
    dA = cubature ? cube(uv->norm(normal(S,uv...))) : sum(dA₄)
    # get centroid
    x = cubature ? cube(uv->S(uv...)*norm(normal(S,uv...)))/dA : sum(x₄ .* dA₄)/dA
    n = cubature ? normalize(cube(uv->normal(S,uv...))) : normalize(sum(n₄.*(wg*wg')))
    # get corners (only for pretty plots)
    xᵤᵥ = S.(u .+ 0.5SA[-du,du], v .+ 0.5SA[-dv,dv]')
    nᵤᵥ = normalize.(normal.(S, u .+ 0.5SA[-du,du], v .+ 0.5SA[-dv,dv]'))
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