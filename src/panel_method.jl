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
ϕ(x,p;G=source,kwargs...) = G==source ? ∫G(x,p) : ∫G(x,p)+∫G(x,reflect(p),G;kwargs...)
reflect(p::NamedTuple) = (x=reflect(p.x),n=reflect(p.n),dA=p.dA,T₁=reflect(p.T₁),T₂=reflect(p.T₂))
reflect(x::SVector{3}) = SA[x[1],x[2],-x[3]]

using ForwardDiff: derivative, gradient, value, partials, Dual
"""
    ∫G(x,p;G=source,kwargs...)

Approximate integral `∫ₚ G(x,x')ds'` over panel `p`. 

An 8x8 quadrature is used when `x==p.x`, otherwise it uses the midpoint.
"""
∫G(x,p,G=source;kwargs...) = x≠p.x ? p.dA*G(x,p.x;kwargs...) : quad8²(x,p)
function ∫G(d::AbstractVector{<:Dual{Tag}},p,G=source;kwargs...) where Tag
    value(d) ≠ p.x && return p.dA*G(d,p.x;kwargs...) # use ∇∫G=∇(_∫G)
    Dual{Tag}(0.,2π*stack(partials.(d))*p.n...)      # enforce ∇∫G(x,x)=2πn̂
end
quad8²(ξ,p;x=xgl8,w=wgl8) = 0.25p.dA*quadgl(x₁->quadgl(x₂->source(ξ,p.x+0.5x₁*p.T₁+0.5x₂*p.T₂);x,w);x,w)

""" 
    ∂ₙϕ(pᵢ,pⱼ;kwargs...) = A

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)::Float64
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