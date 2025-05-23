"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true) -> (x,n̂,dA,x₄,c₄)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, the 
unit normal `n̂=n/|n|` and the surface area `dA≈|n|`, where `n≡T₁×T₂` and the 
tangent vectors are `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`. x₄ are the 2x2 
Gauss-point locations, optionally projected onto the `tangentplane`. c₄
are the panel corners. 
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂;tangentplane=true,signn=1)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = signn*T₁×T₂; mag = hypot(n...); x = S(ξ₁,ξ₂)
    dx = SA[-1/√3,1/√3] # Gauss-points
    x₄ = tangentplane ? ((a,b)->(x+0.5T₁*a+0.5T₂*b)).(dx,dx') : S.(ξ₁ .+ 0.5dξ₁*dx,ξ₂ .+ 0.5dξ₂*dx') 
    (x=x, n=n/mag, dA=mag, x₄=x₄::SMatrix{2,2},c₄=S.(ξ₁ .+ 0.5√3*dξ₁*dx,ξ₂ .+ 0.5√3*dξ₂*dx') )
end

"""
pseudospeed = max(arcspeed,curvespeed) wrapped in an ApproxFun for later analysis
"""
pseudospeed(r,L,c,rng) = max(Fun(arcspeed(r),rng),Fun(curvespeed(r,L,c),rng))
"""
curvespeed = `s' √(L|κ|/8c))`, where `s'≡||r'||` is the arcspeed, `|κ|` is the curvature, 
`L` is the arclength-scale and `c` is the max percent deviation.
"""
curvespeed(r,L,c) = u->√(L*κₙ(r,u)/8c)
arcspeed(r) = u->hypot(derivative(r,u)...)
"""
Normal curvature `κₙ ≡ s'² |κ| = √(||r''||²-(s'')²)`
"""
κₙ(r,u) = √max(0,sum(abs2,derivative(u->derivative(r,u),u))-derivative(arcspeed(r),u)^2)
dist⁻¹(speed,s) = first(roots(cumsum(speed)-s))
linear(a,b,x₀,x₁) = x->(r=(x-x₀)/(x₁-x₀); a*(1-r)+b*r)
"""
    panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,transpose=false,signn=1,kwargs...)

Panelize a parametric `surface` of `u∈[u₀,u₁]` and `v∈[v₀,v₁]` into a `Table` of panels. 

The surface is split into strips roughly `hᵤ` wide, which are then split into panels roughly `hᵥ`
high. Use `transpose=true` to change the strip direction and `signn=-1` to flip the normal direction. 
The parameter `h*c` sets the max devitation of the panel from a straight line by reducing panel size
in regions of high curvature.
"""
function panelize(surface,u₀=0,u₁=1,v₀=0,v₁=1;hᵤ=1,hᵥ=hᵤ,c=0.05,
                  transpose=false,signn=1,verbosecheck=false,kwargs...)
    transpose && return panelize((v,u)->surface(u,v),v₀,v₁,u₀,u₁,;hᵤ=hᵥ,hᵥ=hᵤ,c,
                                 transpose=false,signn=-signn,verbosecheck,kwargs...)

    # Get arcspeed along bottom & top edges as a Fun
    speed₀ = pseudospeed(u->surface(u,v₀),hᵤ,c,u₀..u₁)
    speed₁ = pseudospeed(u->surface(u,v₁),hᵤ,c,u₀..u₁)

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
        speed = pseudospeed(v->surface(u(v),v),hᵥ,c,v₀..v₁)   # speed along strip center
        S = sum(speed); S ≤ 0.5hᵥ && return [] # not enough height
        ve = dist⁻¹.(speed,0:S/round(S/hᵥ):S)  # panel endpoints
        v = 0.5*(ve[2:end]+ve[1:end-1])        # panel centers
        dv = ve[2:end]-ve[1:end-1]             # panel heights

        # Measure max deviation & panel Properties
        verbosecheck && (error = extrema(map(eachindex(v)) do i
                p,a = surface(u(v[i]),v[i]),surface(u(ve[i]),ve[i])
                l = a-surface(u(ve[i+1]),ve[i+1])
                hypot((p-a-l*(p-a)'l/l'l...))/(hᵥ*c)
        end); @show error)
        @. param_props(surface,u(v),v,du(v),dv;signn,kwargs...) 
    end |> Table
end

