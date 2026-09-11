""" source(x,a)

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/norm(x-a)
""" ∫G(x,p) = p.dA*source(x,p.x)

Monopole Green's function for a source panel `p`.
"""
∫G(x,p,args...) = p.dA*source(x,p.x)
""" ∫G(ξ,p,::QuadKernel; d²=25) = ∑ᵢ wgᵢ*source(ξ,xgᵢ)

Gauss quadrature over source panel `p`. Uses a monopole if `r²/dA>d²`. The self-influence integral
is desingularized using the exact tangent plane potential.
"""
function ∫G(ξ,p,::QuadKernel; d²=25,ignore...)
    r²=sum(abs2,ξ-p.x)
    r²>d²*p.dA && return -p.dA/√r²
    r²>0 && return quadgl(x->source(ξ,x),x=p.xg,w=p.wg)
    p.ϕ+(ξ-p.x)'p.v # AD-friendly pre-computed desingularized self-influence
end
""" ∫G(ξ,p,::TriKernel)

Exact integrated potential over a triangular panel. See Katz and Plotkin, "Low-Speed Aerodynamics" (2001)
"""
function ∫G(ξ, p, ::TriKernel; inside= ξ==p.x ? 1 : 0, ignore...)
    r = p.verts .- Ref(ξ); R = norm.(r)
    edges = sum(1:3) do i
        m,t,j = p.inplane[i],p.tangents[i],i%3+1
        numer = R[i] + r[i]'t
        denom = R[j] + r[j]'t
        (numer > 0 && denom > 0) ? r[i]'m * log(numer/denom) : zero(r[i]'m)
    end
    Ω = 2atan(r[1]'*(r[2]×r[3]),prod(R)+sum(i->r[i]'r[i%3+1]*R[(i+1)%3+1],1:3))
    @inbounds edges+r[1]'p.n*Ω-2π*sign(r[1]'p.n)*inside
end
∫G(ξ,p;kwargs...) = ∫G(ξ,p,p.kernel;kwargs...)

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ),0)

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
    a = mapbody!(f,similar(sys.body.q),sys)
    sum(i->a[i]*sys.body.n[i]*sys.body.dA[i],eachindex(a)) # sum quick
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