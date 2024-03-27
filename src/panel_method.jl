"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂) -> (x,n̂,dA,T₁,T₂)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, 
the unit normal `n̂=n/|n|`, the surface area `dA≈|n|`, and the tangent 
vectors `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`, where `n≡T₁×T₂`.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = T₁×T₂; mag = hypot(n...)
    (x=S(ξ₁,ξ₂), n=n/mag, dA=mag, T₁=T₁, T₂=T₂)
end
"""
    ϕ(x,p;G=source,kwargs...)

Disturbance potential of panel `p` on point `x`. 

If the greens function `G≠source` this routine combines the contributions 
of a source at `p` and `G(x,reflect(p))`.
"""
function ϕ(x,p;G=source,kwargs...)
    G==source && return ∫G(x,p)
    return ∫G(x,p)+∫G(x,reflect(p);G,kwargs...)
end
reflect(p::NamedTuple) = (x=reflect(p.x),n=reflect(p.n),dA=p.dA,T₁=reflect(p.T₁),T₂=reflect(p.T₂))
reflect(x::SVector{3}) = SA[x[1],x[2],-x[3]]

"""
    ∫G(x,p;d²=9,G=source,kwargs...)

Approximate integral `∫ₚ G(x,x')ds'` over panel `p`. 

A midpoint quadrature is used when `|x-p.x|² > d²*dA`.
"""
∫G(x,p;kwargs...) = _∫G(x,p;kwargs...)  # wrapper
@fastmath function _∫G(x,p;d²=9,G=source,kwargs...)
    sum(abs2,x-p.x)>d²*p.dA && return p.dA*G(x,p.x;kwargs...) # midpoint quadrature
    x≠p.x && return p.dA*quad2(ξ₁->quad2(ξ₂->G(x,p.x+ξ₁*p.T₁+ξ₂*p.T₂;kwargs...))) # 2²-point
    p.dA*quad8(ξ₁->quad8(ξ₂->G(x,p.x+ξ₁*p.T₁+ξ₂*p.T₂;kwargs...))) # 8²-point
end
function ∫G(d::AbstractVector{<:Dual{Tag}},p) where Tag
    value(d) ≠ p.x && return _∫G(d,p)             # use ∇∫G=∇(_∫G)
    Dual{Tag}(0.,2π*stack(partials.(d))*p.n...)   # enforce ∇∫G(x,x)=2πn̂
end
quad2(f) = 0.5quadgl(x->f(0.5x),x=xgl2,w=wgl2) # integrate over ξ=[-0.5,0.5] with 2 points
quad8(f) = 0.5quadgl(x->f(0.5x),x=xgl8,w=wgl8) # use 8 points instead

""" 
    ∂ₙϕ(pᵢ,pⱼ;kwargs...) = A

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)
Uₙ(pᵢ;U=[1,0,0]) = U ⋅ pᵢ.n
"""
    φ(x,q,panels;kwargs...)

Potential `φ(x) = ∫ₛ q(x')G(x-x')ds' = ∑ᵢqᵢϕ(x,pᵢ)` of `panels` with strengths `q`.
"""
φ(x,q,panels;kwargs...) = sum(qᵢ*ϕ(x,pᵢ;kwargs...) for (qᵢ,pᵢ) in zip(q,panels))
∇φ(x,q,panels;kwargs...) = gradient(x->φ(x,q,panels;kwargs...),x)
body_velocity(q,panels;U=[1,0,0],kwargs...) = map(x->U+∇φ(x,q,panels;kwargs...),panels.x) |> stack
added_mass(q,panels;kwargs...) = sum(p->φ(p.x,q,panels;kwargs...)*p.n*p.dA,panels)
ζ(x,y,q,panels;kwargs...) = derivative(x->φ([x,y,0],q,panels;kwargs...),x)