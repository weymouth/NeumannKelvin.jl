""" source(x,a)

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/norm(x-a)
""" ∫G_kernel(x,p) = p.dA*source(x,p.x)

Monopole Green's function for a source panel `p`.
"""
∫G_kernel(x,p,args...) = p.dA*source(x,p.x)
""" ∫G_kernel(ξ,p,::QuadKernel) = ∑ᵢ wgᵢ*source(ξ,xgᵢ)

Gauss quadrature over source panel `p`.
"""
∫G_kernel(ξ,p,::QuadKernel) = sum(w*source(ξ,x) for (x,w) in zip(p.xg,p.wg))

using ForwardDiff: value, partials, Dual
"""
    ∫G(x,p)

Approximate integral `∫ₚ G(x,x')da'` over source panel `p`. This function enforces ∇∫G(x,x)=2π̂n.
"""
∫G(x,p) = ∫G_kernel(x,p,p.kernel)
function ∫G(d::AbstractVector{<:Dual{Tag,T,N}},p) where {Tag,T,N}
    val = ∫G_kernel(d,p,p.kernel) # use auto-diff
    value(d) ≠ p.x && return val
    ∂ = ntuple(i->2π*sum(j->partials(d[j])[i]*p.n[j],eachindex(d)),N)
    Dual{Tag}(value(val),∂...) # overwrite partials with ∇∫G(x,x)=2πn̂ contribution
end

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ),0.)

"""
    Φ(x,sys)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')da' = ∑ᵢqᵢ∫G(x,pᵢ)` induced by **solved** panel system `sys`.
"""
Φ(x,sys) = sum(m->Φ_dom(x .* m,sys.body),sys.mirrors)
@inline Φ_dom(x,panels) = sum(p->p.q*∫G(x,p),panels)
Φₙ(p,sys) = derivative(t->Φ(p.x+t*p.n,sys),0) # WRT the panel normal
∇Φ(x,sys) = gradient(x′->Φ(x′,sys),x)

"""
    cₚ([x::SVector{3},] sys)

Measure the pressure coefficient cₚ = 1-u²/U², where `U` is the background velocity and
`u = U+∇Φ` is the flow velocity. If no location `x` is given, a vector of cₚ at all body
centers is calculated and is accelerated when Threads.nthreads()>1.

See also: [`Φ`](@ref)
"""
cₚ(x::SVector{3},sys) = 1-sum(abs2,u(x,sys))/sum(abs2,sys.U)
cₚ(sys) = mapbody!(cₚ,similar(sys.body.q),sys)
mapbody!(f,b,sys) = (AK.foreachindex(i-> b[i] = f(sys.body.x[i],sys), b); b)

"""
    u([x::SVector{3},] sys)

Measure the velocity vector `u = U+∇Φ`. If no location `x` is given, a vector of 
u at all body centers is calculated and is accelerated when Threads.nthreads()>1.

See also: [`Φ`](@ref)
"""
u(x::SVector{3},sys) = sys.U+∇Φ(x,sys)
u(sys) = mapbody!(u,similar(sys.body.x),sys)

"""
    steadyforce(sys; S=bodyarea(sys))

Integrated steady pressure force coefficient vector `∫ₛ cₚ nᵢ da/S = Fᵢ/(½ρU²S)`, where `S` is
the body panel area. Computation is accelerated when Threads.nthreads()>1.

See also: [`cₚ`](@ref)
"""
steadyforce(sys;S=bodyarea(sys)) = -surface_integral(cₚ,sys)/S
@inline function surface_integral(f,sys)
    init = neutral = zero(eltype(sys.body.n))
    AK.mapreduce(+, sys.body, AK.get_backend(sys.body.q); init, neutral) do p
        f(p.x,sys) * p.n * p.dA
    end
end

"""
    addedmass(sys; V=bodyvol(sys))

Added mass coefficient force vector `-∫ₛ Φⱼ/Uⱼ nᵢ da/V = mᵢⱼ/ρV` induced by a panel system,
where `V` is the body volume. Computation is accelerated when Threads.nthreads()>1.

**Note:** The index j is set by the velocity vector used to solve the system. For example,
using `U = [0,-1,0]`, means j=2 and addedmass returns the mᵢ₂ vector. Call this function for
j=1:3 to fill in the full added mass matrix,

See also: [`Φ`](@ref)
"""
addedmass(sys;V=bodyvol(sys)) = -surface_integral(Φ,sys)/norm(sys.U)/V
"""
    addedmass(panels::Table)

Convenience function to fill in the full added mass matrix via direct solve.
"""
function addedmass(panels::Table;sys=BodyPanelSystem(panels),V=bodyvol(sys))
    A = influence(sys)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B'              # solution _matrix_ over i=1,2,3
    map(j->addedmass(set_q!(sys,view(Q,:,j));V),1:3) |>stack
end