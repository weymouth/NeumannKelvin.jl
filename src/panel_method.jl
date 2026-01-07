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
∫G(x,p;d²=4,ignore...) = _∫G(x,p;d²)
function ∫G(d::AbstractVector{<:Dual{Tag,T,N}},p;d²=4,kwargs...) where {Tag,T,N}
    val = _∫G(d,p;d²) # use auto-diff
    value(d) ≠ p.x && return val
    ∂ = ntuple(i->2π*sum(j->partials(d[j])[i]*p.n[j],eachindex(d)),N)
    Dual{Tag}(value(val),∂...) # overwrite partials with ∇∫G(x,x)=2πn̂ contribution
end
_∫G(ξ,p;d²) = (!hasproperty(p, :x₄) || sum(abs2,ξ-p.x)>d²*p.dA) ? p.dA*source(ξ,p.x) : quadgl(x->source(ξ,x),x=p.x₄,w=p.w₄)

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)

"""
    PanelSystem(panels, q=0.; ϕ=∫G, sym_axes=(), kwargs...)

Represents a panel **system**, i.e., a set of `panels` with strengths `q` used to
satisfy the boundary conditions for the Green's function `ϕ(x,p;kwargs...)`.
The strength `q` is added as a column to `panels`.

Setting `sym_axes` imposes symmetry conditions on the solution using the method of
images. For example, `sym_axes=(2,3)` mirrors each contribution across `y=0,z=0`, so
only one quarter of a centered & symmetric geometry needs to be covered in panels.

# Usage
```julia
sys = PanelSystem(panels; sym_axes=2) # half a body with y-symmetry
gmressolve!(sys, atol=1e-6)           # approximate solve - but still O(N²) operations!
extrema(cₚ(sys))                      # check solution quality
```
"""
struct PanelSystem{T,P,M,K}
    panels::T
    ϕ::P
    mirrors::M
    kwargs::K
end
function PanelSystem(panels, q=sim_fill(0.,panels.dA); ϕ=∫G, sym_axes=(), kwargs...)
    PanelSystem(Table(panels;q), ϕ, mirrors(sym_axes...), kwargs) #, Dict(kwargs...)) # SUPER slow
end
sim_fill(val,array::AbstractArray) = (a = similar(array,typeof(val)); a .= val; a)
@inline function mirrors(axes...)
    M = length(axes)
    ntuple(combo -> SA[ntuple(i -> any(j -> axes[j] == i && ((combo >> (j-1)) & 1) == 1, 1:M) ? -1 : 1, 3)...], 1 << M)
end

# Pretty printing
Base.show(io::IO, sys::PanelSystem) = print(io, "PanelSystem($(length(sys.panels)) panels")
Base.show(io::IO, ::MIME"text/plain", sys::PanelSystem) = abstract_show(io,sys)
function abstract_show(io,sys)
    show(io,sys);println()
    println(io, "  total body area: $(body_area(sys))")
    println(io, "  mirrors: $(sys.mirrors)")
    println(io, "  kwargs: $(sys.kwargs...)")
    println(io, "  strength extrema: $(extrema(sys.panels.q))")
end
body_area(sys) = sum(sys.panels.dA)

"""
    Φ(x,sys)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')da' = ∑ᵢqᵢ∫G(x,pᵢ)` induced by **solved** panel system `sys`.

See also: [`PanelSystem`](@ref)
"""
Φ(x,sys;kwargs...) = sum(Φ_sys(x .* m,sys;kwargs...) for m in sys.mirrors)
@inline Φ_sys(x,sys;ignore...) = sum(pᵢ.q*∫G(x,pᵢ;sys.kwargs...) for pᵢ in sys.panels)
Φₙ(p,sys;kwargs...) = derivative(t->Φ(p.x+t*p.n,sys;kwargs...),0) # WRT the panel normal
Φₓ(x,sys;kwargs...) = derivative(t->Φ(x+t*SA[1,0,0],sys;kwargs...),0)
∇Φ(x,sys) = gradient(x′->Φ(x′,sys),x)

"""
    cₚ([x::SVector{3},] sys; U=SVector(-1,0,0))

Measure the pressure coefficient cₚ = 1-u²/U², where `U` is the free stream velocity and
`u = U+∇Φ` is the flow velocity. If no location `x` is given, a vector of cₚ at all panel
centers is returned. Computation is accelerated when Threads.nthreads()>1 and/or when using
a solved Barnes-Hut panel tree.

See also: [`Φ`](@ref)
"""
cₚ(x::SVector{3},sys;U=SVector(-1,0,0)) = 1-sum(abs2,U+∇Φ(x,sys))/sum(abs2,U)
function cₚ(sys;U=SVector(-1,0,0))
    b = similar(sys.panels.q)
    AK.foreachindex(b) do i
        b[i] = cₚ(sys.panels.x[i],sys;U)
    end; b
end

"""
    steadyforce(sys;U=SVector(-1,0,0))

Integrated steady pressure force coefficient vector `∫ₛ cₚ nᵢ da/S = Fᵢ/(½ρU²S)`, where `S` is
the body panel area. Computation is accelerated when Threads.nthreads()>1 and/or when using a
solved Barnes-Hut panel tree.

See also: [`cₚ`](@ref)
"""
steadyforce(sys;U=SVector(-1,0,0)) = surface_integral((x,sys)->cₚ(x,sys;U),sys)/body_area(sys)
@inline function surface_integral(f,sys)
    init = neutral = zero(eltype(sys.panels.n))
    AK.mapreduce(+, sys.panels, AK.get_backend(sys.panels.q); init, neutral) do p
        f(p.x,sys) * p.n * p.dA
    end
end

"""
    addedmass(sys)

Added mass coefficient force vector `-∫ₛ Φⱼ nᵢ da = mᵢⱼ/ρV` induced by a panel system solved
with unit velocity in direction j, ie `j=2` requires `U=[0,±1,0]`. Computation is accelerated
when Threads.nthreads()>1 and/or when using a solved Barnes-Hut panel tree.

**Note:** Call this function for j=1:3 to fill in the full added mass matrix,

See also: [`Φ`](@ref)
"""
addedmass(sys) = -surface_integral(Φ,sys)

"""
    addedmass(panels::Table,kwargs...)

Convenience function to fill in the full added mass matrix via direct solve.
"""
function addedmass(panels::Table;kwargs...)
    A = ∂ₙϕ.(panels,panels';kwargs...)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B'              # solution _matrix_ over i=1,2,3
    map(j->addedmass(PanelSystem(panels,view(Q,:,j);kwargs...)),1:3) |>stack
end