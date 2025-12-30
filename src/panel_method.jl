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
function ∫G(d::AbstractVector{<:Dual{Tag,T,N}},p;d²=4,kwargs...) where {Tag,T,N}
    val = _∫G(d,p;d²) # use auto-diff
    value(d) ≠ p.x && return val
    ∂ = ntuple(i->2π*sum(j->partials(d[j])[i]*p.n[j],eachindex(d)),N)
    Dual{Tag}(value(val),∂...) # overwrite partials with ∇∫G(x,x)=2πn̂ contribution
end
_∫G(ξ,p;d²) = (!hasproperty(p,:x₄) || sum(abs2,ξ-p.x)>d²*p.dA) ? p.dA*source(ξ,p.x) : quadgl(x->source(ξ,x),x=p.x₄,w=p.w₄)

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G,kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)

abstract type AbstractPanelSystem end
"""
    PanelSystem(panels, q=zeros; ϕ=∫G, kwargs...)

Represents a panel **system**, i.e., a set of `panels` with strengths `q` used to
statisfy the boundary conditions for the given Green's function `ϕ(x,p;kwargs...)`.

# Usage
```julia
# You can solve the system first directly...
kwargs = (ϕ=∫G,d²=Inf)              # set the potential and keyword arguments
A = ∂ₙϕ.(panels,panels';kwargs...)  # create the influence matrix (N² elements!)
b = components(panels.n,1)          # and right-hand vector
q = A\\b                             # direct solve for the strength

# .. and then bundle everything together to measure.
sys1 = PanelSystem(panels, q; kwargs...)
extrema(panel_cp(sys1))      # for example, the extreme values of cₚ

# Or you can bundle first and use an indirect solver
sys2 = PanelSystem(panels; kwargs...)
GMRESsolve!(sys2,atol=1e-6)  # approximate solve (but still O(N²) operations!)
extrema(panel_cp(sys2))      # should match with tol ~ 1e-6
```
"""
struct PanelSystem{T,F,K} <: AbstractPanelSystem
    panels::T # includes strengths!
    ϕ::F
    kwargs::K
end
PanelSystem(panels,q;ϕ=∫G,kwargs...) = PanelSystem(Table(panels;q),ϕ,kwargs)
PanelSystem(panels;ϕ=∫G,kwargs...) = (q=similar(panels.dA); q.=0; PanelSystem(panels,q;ϕ,kwargs))

# Pretty printing
Base.show(io::IO, sys::PanelSystem) = print(io, "PanelSystem($(length(sys.panels)) panels")
Base.show(io::IO, ::MIME"text/plain", sys::AbstractPanelSystem) = abstract_show(io,sys)
function abstract_show(io,sys)
    show(io,sys);println()
    println(io, "  total area: $(total_area(sys))")
    println(io, "  strength extrema: $(extrema(sys.panels.q))")
end
total_area(sys) = sum(sys.panels.dA)

"""
    Φ(x,sys)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')da' = ∑ᵢqᵢϕ(x,pᵢ)` induced by **solved** panel system `sys`.

See also: [`PanelSystem`](@ref)
"""
Φ(x,sys;ignore...) = sum(pᵢ.q*sys.ϕ(x,pᵢ;sys.kwargs...) for pᵢ in sys.panels)
∇Φ(x,sys) = gradient(x′->Φ(x′,sys),x)

"""
    panel_cp(sys;U=SVector(-1,0,0)) -> cₚ

Measure the pressure coefficient `cₚ = 1-u²/U²` on each panel center, where `U` is the free stream
velocity and `u = U+∇Φ` is the flow velocity. Computation is accelerated when Threads.nthreads()>1
and/or when using a solved Barnes-Hut panel tree.

See also: [`Φ`](@ref)
"""
panel_cp(sys;U=SVector(-1,0,0)) = (b=similar(sys.panels.q);panel_cp!(b,sys;U);b)
panel_cp!(b,sys;U=SVector(-1,0,0)) = AK.foreachindex(b) do i
    b[i] = local_cp(sys.panels.x[i],sys;U)
end
local_cp(x,sys;U=SVector(-1,0,0)) = 1-sum(abs2,U+∇Φ(x,sys))/sum(abs2,U)

"""
    steady_force(sys;U=SVector(-1,0,0))

Integrated steady pressure force coefficient vector `∫ₛ cₚ nᵢ da/A = Fᵢ/(½ρU²A)`, where `A`
is the total panel area. Computation is accelerated when Threads.nthreads()>1 and/or when using
a solved Barnes-Hut panel tree.

See also: [`Φ`](@ref)
"""
steady_force(sys;U=SVector(-1,0,0)) = surface_integral(sys,(x,sys)->local_cp(x,sys;U))/total_area(sys)
@inline function surface_integral(sys, f)
    panels = sys.panels
    init = neutral = zero(eltype(panels.n))
    AK.mapreduce(+, panels, AK.get_backend(panels.q); init, neutral) do p
        f(p.x,sys) * p.n * p.dA
    end
end

"""
    added_mass(sys)

Added mass coefficient force vector `-∫ₛ Φⱼ nᵢ da = mᵢⱼ/ρV` induced by a panel system
solved with unit velocity in direction j, ie `j=2` requires `U=[0,1,0]`.

**Note:** Call this function for j=1:3 to fill in the full added mass matrix,

See also: [`Φ`](@ref)
"""
added_mass(sys) = -surface_integral(sys,Φ)

"""
    added_mass(panels::Table,kwargs...)

Convenience function to fill in the full added mass matrix via direct solve.
"""
function added_mass(panels::Table;kwargs...)
    A = ∂ₙϕ.(panels,panels';kwargs...)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B'              # solution _matrix_ over i=1,2,3
    map(j->added_mass(PanelSystem(panels,view(Q,:,j);kwargs...)),1:3) |>stack
end