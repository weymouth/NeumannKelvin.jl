"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂) -> (x,n̂,dA,T₁,T₂)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, 
the unit normal `n̂=n/|n|`, the surface area `dA≈|n|`, and the tangent 
vectors `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`, where `n≡T₁×T₂`.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = T₁×T₂; mag = hypot(n...); x = S(ξ₁,ξ₂)
    x₄ = @SMatrix [x+0.5T₁*x₁+0.5T₂*x₂ for x₁ in xgl2, x₂ in xgl2]
    (x=x, n=n/mag, dA=mag, x₄=x₄)
end
"""
    ϕ(x,p;G=source,kwargs...)

Disturbance potential of panel `p` on point `x`. 

If `G≠source` this routine includes the `reflect(p)` contributions.
"""
function ϕ(x,p;G=source,kwargs...) 
    G==source && return ∫G(x,p;kwargs...)
    ∫G(x,p;kwargs...)-∫G(x,reflect(p);kwargs...)+p.dA*G(x,reflect(p.x);kwargs...)
end
reflect(p::NamedTuple) = (x=reflect(p.x),n=reflect(p.n),dA=p.dA,x₄=reflect.(p.x₄))
reflect(x::SVector{3}) = SA[x[1],x[2],-x[3]]

using ForwardDiff: derivative, gradient, value, partials, Dual
"""
    ∫G(x,p;d²=4)

Approximate integral `∫ₚ G(x,x')ds'` over source panel `p`. 

A 2x2 quadrature is used when `|x-p.x|²≤d²p.dA`, otherwise it uses the midpoint.
"""
∫G(x,p;d²=4,kwargs...) = _∫G(x,p;d²)
function ∫G(d::AbstractVector{<:Dual{Tag}},p;d²=4,kwargs...) where Tag
    value(d) ≠ p.x && return _∫G(d,p;d²) # use auto-diff
    Dual{Tag}(0.,2π*stack(partials.(d))*p.n...) # enforce ∇∫G(x,x)=2πn̂
end
_∫G(ξ,p;d²) = sum(abs2,ξ-p.x)>d²*p.dA ? p.dA*source(ξ,p.x) : 0.25p.dA*sum(source(ξ,x) for x in p.x₄)
""" 
    ∂ₙϕ(pᵢ,pⱼ;kwargs...) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)::Float64

""" 
   influence(panels;kwargs...) = ∂ₙϕ.(panels,panels';kwargs...) = A

Normal velocity influence matrix. Computation is accelerated with 
multi-threading when `Threads.nthreads()>1`.
"""
influence(panels;kwargs...) = influence!(Array{Float64}(undef,length(panels),length(panels)),panels;kwargs...)
function influence!(A,panels;kwargs...)
    isfirstcall[] && ∂ₙϕ(panels[1],panels[end];kwargs...) # initialize once
    ThreadsX.foreach(CartesianIndices(A)) do I
        A[I] = ∂ₙϕ(panels[I[1]],panels[I[2]];kwargs...)   # multi-thread fill
    end; A
end
"""
    φ(x,q,panels;kwargs...)

Potential `φ(x) = ∫ₛ q(x')G(x-x')ds' = ∑ᵢqᵢϕ(x,pᵢ)` of `panels` with strengths `q`.
"""
φ(x,q,panels;kwargs...) = ThreadsX.sum(qᵢ*ϕ(x,pᵢ;kwargs...) for (qᵢ,pᵢ) in zip(q,panels))
∇φ(x,q,panels;kwargs...) = gradient(x->φ(x,q,panels;kwargs...),x)
ζ(x,y,q,panels;kwargs...) = derivative(x->φ(SVector(x,y,0),q,panels;kwargs...),x)

"""
    steady_force(q,panels,U=SVector(-1,0,0);kwargs...)

Integrated pressure force coefficient Cₚ =∫ₛ cₚ n da of `panels` with strengths `q`.
where cₚ = 1-u²/U², `U` is the freestreeam velocity and u=U+∇φ is the flow velocity.
"""
steady_force(q,panels;U=SVector(-1,0,0),kwargs...) = ThreadsX.sum(panels) do pᵢ
    u² = sum(abs2,U+∇φ(pᵢ.x,q,panels;kwargs...))
    cₚ = 1-u²/sum(abs2,U)
    cₚ*pᵢ.n*pᵢ.dA
end
"""
    added_mass(panels;kwargs...)

Added mass matrix mᵢⱼ = -∫ₛ φᵢ(x) nⱼda for a set of `panels`, 
where φᵢ is the potential due to unit motion in direction i.
"""
function added_mass(panels;kwargs...)
    A = influence(panels;kwargs...)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B' # solution _matrix_ over i=1,2,3
    [-ThreadsX.sum(p->φ(p.x,view(Q,:,i),panels;kwargs...)*p.n[j]*p.dA,panels) for i in 1:3, j in 1:3]
end