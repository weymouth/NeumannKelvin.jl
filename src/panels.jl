"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true) -> (x,n̂,dA,x₄,⛶)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, the 
unit normal `n̂=n/|n|` and the surface area `dA≈|n|`, where `n≡T₁×T₂` and the 
tangent vectors are `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`. x₄ are the 2x2 
Gauss-point locations, optionally projected onto the `tangentplane`. 
The panel corners ⛶ are only used for panel visualization & assessment.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true,signn=1)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = signn*T₁×T₂; mag = hypot(n...); x = S(ξ₁,ξ₂)
    dx = SA[-1/√3,1/√3] # Gauss-points
    x₄ = tangentplane ? ((a,b)->(x+0.5T₁*a+0.5T₂*b)).(dx,dx') : S.(ξ₁ .+ 0.5dξ₁*dx,ξ₂ .+ 0.5dξ₂*dx') 
    (x=x, n=n/mag, dA=mag, x₄=x₄::SMatrix{2,2},⛶=S.(ξ₁ .+ dξ₁*SA[-0.5,0.5],ξ₂ .+ dξ₂*SA[-0.5,0.5]'))
end
"""
    deviation = distance from panel center to plane defined by the corners
"""
function deviation(p)
    a =   0.5p.⛶[1,1]+0.5p.⛶[1,2]    # plane base
    l = a-0.5p.⛶[2,1]-0.5p.⛶[2,2]    # vector to plane top
    hypot((p.x-a-l*(p.x-a)'l/l'l...)) # distance from center
end

using QuadGK,DataInterpolations
"""
    arclength(r,Δs,c,low,high) -> S,(s⁻¹(s)->u)

Find the pseudo-arclength `S` along a curve `r(u), u ∈ [low,high]` 
by integrating the pseudo-arcspeed `s' = max(l',√(Δs*κₙ/8c)))` where 
`l' ≡ ∥r'∥` is the arcspeed and `κₙ` is the normal curvature. 
The inverse function `s⁻¹(s) = ∫₀ˢ du/ds = u` is also returned.

The speed `s'` is defined such that `S/Δs` equal length segments 
along `r` deviate `δ≤Δs*c` from the curve. Starting from `δ≈Δl²/8κ`,
`Δl≈l'*Δu` and `Δs≈s'*Δu` gives `s'≥l'√(Δs*κ/8c)`. Demanding `s'≥l'` 
and substituting `κₙ = l'² κ` gives the rate equation above.
"""
function arclength(r,Δs,c,low,high)
    # Use quadgk to adaptively sample ∫ds, returning S and subintervals Δᵢ
    @inline speed(u) = max(arcspeed(r)(u),√(Δs*κₙ(r,u)/8c))
    S,_,Δᵢ = quadgk_segbuf(speed,low,high,rtol=1e-5,order=3)
    # Order the subintervals and accumulate the arclength data s(uᵢ)=∑ⁱⱼ₌₀ Δsⱼ
    sort!(Δᵢ,by=Δ->Δ.a)
    u,s = [low; map(Δ->Δ.b,Δᵢ)],[zero(S); cumsum(map(Δ->Δ.I,Δᵢ))]
    # Get the bounded derivative data and construct a Monotonic Hermite Cubic Spline for s⁻¹
    du = map(eachindex(u)) do i
        min(3secant(Δᵢ[max(1,i-1)]),3secant(Δᵢ[min(i,length(Δᵢ))]),inv(speed(i==1 ? Δᵢ[1].a : Δᵢ[i-1].b)))
    end
    S,CubicHermiteSpline(du,u,s,extrapolation = ExtrapolationType.Constant)
end
arcspeed(r) = u->hypot(derivative(r,u)...)
κₙ(r,u) = √max(0,sum(abs2,derivative(u->derivative(r,u),u))-derivative(arcspeed(r),u)^2)
secant(Δ)=(Δ.b-Δ.a)/Δ.I
linear(a,b,x₀,x₁) = x->(r=(x-x₀)/(x₁-x₀); a*(1-r)+b*r)
"""
    panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,signn=1,kwargs...)

Panelize a parametric `surface` of `u∈[u₀,u₁]` and `v∈[v₀,v₁]` into a `Table` of panels. 

The surface is split into strips roughly `hᵤ` wide, which are split into panels roughly `hᵥ` high. 
Use `transpose=true` to change the strip direction and `signn=-1` to flip the normal direction. 
The parameter `(hᵤ+hᵥ)*c` sets the max deviation of the panel from a flat plane by reducing 
panel size in regions of high curvature.
"""
function panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,signn=1,kwargs...)
    transpose && return panelize((v,u)->surface(u,v),v₀,v₁,u₀,u₁,;hᵤ=hᵥ,hᵥ=hᵤ,c,transpose=false,signn=-signn,kwargs...)
    # Check inputs and get output type
    @assert u₀<u₁ && v₀<v₁ && hᵤ>0 && hᵥ>0 && c>0 && abs(signn)==1 "Invalid panelize inputs"
    p = param_props(surface,0.5u₀+0.5u₁,u₁-u₀,0.5v₀+0.5v₁,v₁-v₀;kwargs...)

    # Get arcslength and inverse along bottom & top edges
    S₀,s₀⁻¹ = arclength(u->surface(u,v₀),hᵤ,c,u₀,u₁)
    S₁,s₁⁻¹ = arclength(u->surface(u,v₁),hᵤ,c,u₀,u₁)

    # Set number of strips
    min(S₀,S₁) ≤ 0.5hᵤ && return nothing # not enough width
    N = Int(round((S₀+S₁)/2hᵤ))          # number of strips

    # Find equidistant points along bottom & top edges
    u₀ᵢ,u₁ᵢ = s₀⁻¹(range(0,S₀,N+1)),s₁⁻¹(range(0,S₁,N+1))

    # Mapreduce across strips
    mapreduce(vcat,1:N, init = Vector{typeof(p)}()) do i
        u = linear(0.5(u₀ᵢ[i+1]+u₀ᵢ[i]),0.5(u₁ᵢ[i+1]+u₁ᵢ[i]),v₀,v₁) # Parametric center
        du = linear(u₀ᵢ[i+1]-u₀ᵢ[i],u₁ᵢ[i+1]-u₁ᵢ[i],v₀,v₁)          # Parametric width

        # Find equidistant points along strip
        S,s⁻¹ = arclength(v->surface(u(v),v),hᵥ,c,v₀,v₁) # speed along strip center
        S ≤ 0.5hᵥ && return []          # not enough height
        ve = s⁻¹(0:S/round(S/hᵥ):S)     # panel endpoints
        v = 0.5*(ve[2:end]+ve[1:end-1]) # panel centers
        dv = ve[2:end]-ve[1:end-1]      # panel heights

        # Measure panel Properties
        @. param_props(surface,u(v),v,du(v),dv;signn,kwargs...) 
    end |> Table
end