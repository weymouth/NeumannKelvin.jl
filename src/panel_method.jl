""" source(x,a)

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/norm(x-a)
""" ∫G_kernel(x,p) = p.dA*source(x,p.x)

Monopole Green's function for a source panel `p`.
"""
∫G_kernel(x,p,args...) = p.dA*source(x,p.x)
""" ∫G_kernel(ξ,p,::QuadKernel;d²=4) = ∑ᵢ wgᵢ*source(ξ,xgᵢ)

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

See also: [`PanelSystem`](@ref)
"""
Φ(x,sys) = sum(Φ_sys(x .* m,sys) for m in sys.mirrors)
@inline function Φ_sys(x,(;panels)::AbstractPanelSystem)
    val = zero(eltype(x))
    @simd for pᵢ in panels
        val += pᵢ.q*∫G(x,pᵢ)
    end; val
end
Φₙ(p,sys) = derivative(t->Φ(p.x+t*p.n,sys),0) # WRT the panel normal
Φₓ(x,sys) = derivative(t->Φ(x+t*SA[1,0,0],sys),0)
∇Φ(x,sys) = gradient(x′->Φ(x′,sys),x)

"""
    ζ([x::SVector{3},] sys; U=1)

Scaled linear free surface elevation `ζ/ℓ=Φₓ/U` induced by **solved** panel system `sys`.
If no location `x` is given, a vector of ζ at all freesurf centers is returned.

See also: [`Φ`](@ref)
"""
ζ(x::SVector{3},sys;U=1) = Φₓ(x,sys)/abs(U)
function ζ(sys;U=1)
    b = similar(sys.freesurf.q)
    AK.foreachindex(b) do i
        b[i] = ζ(sys.freesurf.x[i],sys;U)
    end; b
end

"""
    cₚ([x::SVector{3},] sys; U=SVector(-1,0,0))

Measure the pressure coefficient cₚ = 1-u²/U², where `U` is the free stream velocity and
`u = U+∇Φ` is the flow velocity. If no location `x` is given, a vector of cₚ at all body
centers is returned. Computation is accelerated when Threads.nthreads()>1 and/or when using
a solved Barnes-Hut panel tree.

See also: [`Φ`](@ref)
"""
cₚ(x::SVector{3},sys;U=SVector(-1,0,0)) = 1-sum(abs2,U+∇Φ(x,sys))/sum(abs2,U)
function cₚ(sys;U=SVector(-1,0,0))
    b = similar(sys.body.q)
    AK.foreachindex(b) do i
        b[i] = cₚ(sys.body.x[i],sys;U)
    end; b
end

"""
    steadyforce(sys;U=SVector(-1,0,0))

Integrated steady pressure force coefficient vector `∫ₛ cₚ nᵢ da/S = Fᵢ/(½ρU²S)`, where `S` is
the body panel area. Computation is accelerated when Threads.nthreads()>1 and/or when using a
solved Barnes-Hut panel tree.

See also: [`cₚ`](@ref)
"""
steadyforce(sys;U=SVector(-1,0,0)) = surface_integral((x,sys)->cₚ(x,sys;U),sys)/bodyarea(sys)
@inline function surface_integral(f,sys)
    body = sys.body
    init = neutral = zero(eltype(body.n))
    AK.mapreduce(+, body, AK.get_backend(body.q); init, neutral) do p
        f(p.x,sys) * p.n * p.dA
    end
end

"""
    addedmass(sys; Uⱼ=-1, V=bodyvol(sys))

Added mass coefficient force vector `-∫ₛ Φⱼ/Uⱼ nᵢ da/V = mᵢⱼ/ρV` induced by a panel system,
where `V` is the body volume. Computation is accelerated when Threads.nthreads()>1.

**Note:** The index j is set by the velocity vector used to solve the system. For example,
using `U = [0,-1,0]`, means j=2 and addedmass returns the mᵢ₂ vector. Call this function for
j=1:3 to fill in the full added mass matrix,

See also: [`Φ`](@ref)
"""
addedmass(sys;Uⱼ=-1,V=bodyvol(sys)) = -surface_integral(Φ,sys)/abs(Uⱼ)/V
"""
    addedmass(panels::Table)

Convenience function to fill in the full added mass matrix via direct solve.
"""
function addedmass(panels::Table;sys=PanelSystem(panels),V=bodyvol(sys))
    A = ∂ₙϕ.(panels,panels')
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B'              # solution _matrix_ over i=1,2,3
    map(j->addedmass(set_q!(sys,view(Q,:,j));V),1:3) |>stack
end