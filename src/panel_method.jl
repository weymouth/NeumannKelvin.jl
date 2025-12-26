""" source(x,a)

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/norm(x-a)

using ForwardDiff: value, partials, Dual
"""
    ∫G(x,p;d²=4)

Approximate integral `∫ₚ G(x,x')da'` over source panel `p`.

A 2x2 quadrature is used when `|x-p.x|²≤d²p.dA`, otherwise it uses the midpoint.
"""
∫G(x,p;d²=4,kwargs...) = _∫G(x,p;d²)
function ∫G(d::AbstractVector{<:Dual{Tag}},p;d²=4,kwargs...) where Tag
    value(d) ≠ p.x && return _∫G(d,p;d²) # use auto-diff
    Dual{Tag}(0.,2π*stack(partials.(d))*p.n...) # enforce ∇∫G(x,x)=2πn̂
end
_∫G(ξ,p;d²) = (!hasproperty(p,:x₄) || sum(abs2,ξ-p.x)>d²*p.dA) ? p.dA*source(ξ,p.x) : quadgl(x->source(ξ,x),x=p.x₄,w=p.w₄)

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)

"""
   influence(panels;kwargs...) = ∂ₙϕ.(panels,panels';kwargs...) = A

Normal velocity influence matrix.
Computation is accelerated with multi-threading when `Threads.nthreads()>1`.
"""
influence(panels;kwargs...) = influence!(Array{Float64}(undef,length(panels),length(panels)),panels;kwargs...)
function influence!(A,panels;kwargs...)
    isfirstcall[] && ∂ₙϕ(panels[1],panels[end];kwargs...) # initialize once
    ThreadsX.foreach(CartesianIndices(A)) do I
        A[I] = ∂ₙϕ(panels[I[1]],panels[I[2]];kwargs...)   # multi-thread fill
    end; A
end
"""
    Φ(x,q,panels;ϕ=∫G,kwargs...)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')ds' = ∑ᵢqᵢϕ(x,pᵢ)` of `panels` with strengths `q`.
Computation is accelerated with multi-threading when `Threads.nthreads()>1`.
"""
Φ(x,q,panels;ϕ=∫G,kwargs...) = ThreadsX.sum(qᵢ*ϕ(x,pᵢ;kwargs...) for (qᵢ,pᵢ) in zip(q,panels))
∇Φ(x,q,panels;kwargs...) = gradient(x->Φ(x,q,panels;kwargs...),x)
ζ(x,y,q,panels;kwargs...) = derivative(x->Φ(SVector(x,y,0),q,panels;kwargs...),x)

"""
    steady_force(q,panels,U=SVector(-1,0,0);kwargs...)

Integrated pressure force coefficient Cₚ =∫ₛ cₚ n da of `panels` with strengths `q`.
where cₚ = 1-u²/U², `U` is the freestreeam velocity and u=U+∇Φ is the flow velocity.
Computation is accelerated with multi-threading when `Threads.nthreads()>1`.
"""
steady_force(q,panels;U=SVector(-1,0,0),kwargs...) = ThreadsX.sum(panels) do pᵢ
    u² = sum(abs2,U+∇Φ(pᵢ.x,q,panels;kwargs...))
    cₚ = 1-u²/sum(abs2,U)
    cₚ*pᵢ.n*pᵢ.dA
end
"""
    added_mass(panels;kwargs...)

Added mass matrix mᵢⱼ = -∫ₛ Φᵢ(x) nⱼ da for a set of `panels`, where Φᵢ is the
potential due to unit motion in direction i.
Computation is accelerated with multi-threading when `Threads.nthreads()>1`.
"""
function added_mass(panels;kwargs...)
    A = influence(panels;kwargs...)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B' # solution _matrix_ over i=1,2,3
    [-ThreadsX.sum(p->Φ(p.x,view(Q,:,i),panels;kwargs...)*p.n[j]*p.dA,panels) for i in 1:3, j in 1:3]
end