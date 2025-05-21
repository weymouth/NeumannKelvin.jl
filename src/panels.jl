"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true) -> (x,n̂,dA,x₄)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, the 
unit normal `n̂=n/|n|` and the surface area `dA≈|n|`, where `n≡T₁×T₂` and the 
tangent vectors are `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`. x₄ are the 2x2 
Gauss-point locations, optionally projected onto the `tangentplane`.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true,signn=1)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = signn*T₁×T₂; mag = hypot(n...); x = S(ξ₁,ξ₂)
    dx = SA[-1/√3,1/√3]; x₄ = S.(ξ₁ .+ 0.5dξ₁*dx,ξ₂ .+ 0.5dξ₂*dx') # Gauss-points
    tangentplane && (x₄ = ((a,b)->(x+0.5T₁*a+0.5T₂*b)).(dx,dx'))
    (x=x, n=n/mag, dA=mag, x₄=x₄::SMatrix{2,2})
end

"""
Sets a local pseudo-arcspeed 

`s̃' = max(s',√(ϵ|s''|/8))`

where `s'=|d curve/du|` is the arcspeed, `s''` is the acceleration and `ϵ` is a distance-scale. 
In regions of large curvature `|κ| ∝ s''/s'²`, the second term will activate. Integrating `s̃'` in 
such a region gives `Δs̃=Δs √(ϵ|κ|/8)` meaning our panel's true arclength Δs is scaled down
proportionally to √(ϵ|κ|/8), ensuring the deviation along the Δs arc is bounded by ϵ.
"""
pseudospeed(curve,ϵ) = u->max(hypot(derivative(curve,u)...),√(ϵ*hypot(derivative(u->derivative(curve,u),u)...)/8))
dist⁻¹(speed,s) = first(roots(cumsum(speed)-s))
linear(a,b,x₀,x₁) = x->(r=(x-x₀)/(x₁-x₀); a*(1-r)+b*r)
"""
    panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,signn=1,kwargs...)

Panelize a parametric `surface` of `u∈[u₀,u₁]` and `v∈[v₀,v₁]` into a `Table` of panels. 

The surface is split into strips roughly `hᵤ` wide, which are then split into panels roughly `hᵥ`
high. Use `transpose=true` to change the strip direction and `signn=-1` to flip the normal direction. 
The parameter `c` sets the max percent devitation of the panel from a straight line, clustering panels
in regions of high curvature.
"""
function panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,signn=1,kwargs...)
    transpose && return equiarea_panels((v,u)->surface(u,v),v₀,v₁,u₀,u₁,;hᵤ=hᵥ,hᵥ=hᵤ,transpose=false,signn=-signn,kwargs...)

    # Get arcspeed along bottom & top edges as a Fun
    speed₀ = Fun(pseudospeed(u->surface(u,v₀),hᵤ/c),u₀..u₁)
    speed₁ = Fun(pseudospeed(u->surface(u,v₁),hᵤ/c),u₀..u₁)

    # Get arclength (integral of speed) and number of strips
    S₀,S₁ = sum(speed₀),sum(speed₁)
    min(S₀,S₁) ≤ 0.5hᵤ && return nothing # not enough width
    N = Int(round((S₀+S₁)/2hᵤ))          # number of strips

    # Find equidistant points along bottom & top edges
    u₀ᵢ = dist⁻¹.(speed₀,range(0,S₀,N+1)) 
    u₁ᵢ = dist⁻¹.(speed₁,range(0,S₁,N+1))

    # Mapreduce across strips
    mapreduce(vcat,1:N) do i
        # Define index functions for strip i
        u = linear(0.5(u₀ᵢ[i+1]+u₀ᵢ[i]),0.5(u₁ᵢ[i+1]+u₁ᵢ[i]),v₀,v₁) # Parametric center
        du = linear(u₀ᵢ[i+1]-u₀ᵢ[i],u₁ᵢ[i+1]-u₁ᵢ[i],v₀,v₁)          # Parametric width

        # Find equidistant points along strip
        speed = Fun(pseudospeed(v->surface(u(v),v),hᵥ/c),v₀..v₁)   # speed along strip center
        S = sum(speed); S ≤ 0.5hᵥ && return []        # not enough height
        ve = dist⁻¹.(speed,0:S/round(S/hᵥ):S)         # panel endpoints
        v = 0.5*(ve[2:end]+ve[1:end-1])               # panel centers
        dv = ve[2:end]-ve[1:end-1]                    # panel heights

        # Measure panels 
        @. param_props(surface,u(v),v,du(v),dv;signn,kwargs...) 
    end |> Table
end
