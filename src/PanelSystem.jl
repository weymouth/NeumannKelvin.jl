"""
    PanelSystem(boundaries; domain=Unbounded(), U=SA[-1,0,0])

Represents a panel **system**, i.e., the `boundaries` of a fluid `domain` covered in panels
with strengths `q` used to satisfy each boundary's condition.

- `boundaries`: A body (`Table` or `PanelTree`), `FreeSurface`, or a tuple of these.
- `domain`: `Unbounded` or `Kelvin` fluid domain.
- `U::SVector{3}`: Optional background flow vector.

**Note** all body panel normals must point *into* the fluid region.

# Usage
```julia
sys = PanelSystem(body_panels) # body only
gmressolve!(sys, atol=1e-6)    # approximate solve - but still O(N²) operations!
extrema(cₚ(sys))               # check solution quality
```
"""
struct PanelSystem{B<:Tuple,D<:FluidDomain,T}
    boundaries::B
    domain::D
    U::SVector{3,T}
end
PanelSystem(boundary;kwargs...) = PanelSystem((boundary,);kwargs...)
function PanelSystem(boundaries::Tuple;domain=Unbounded(),U=SA[-1,0,0])
    allequal(map(b->eltype(b.dA),boundaries)) || throw(ArgumentError("Floating point type of all panels must match"))
    sys = PanelSystem(map(withq,boundaries),domain,U)
    check(domain,sys); foreach(b->check(b,sys),boundaries); sys
end
withq(table::Table) = Table(table; q=zeros_like(table.dA))
withq(boundary) = boundary
check(domain_or_boundary,sys) = nothing

Base.show(io::IO,table::Table) = println(io,"Table with $(length(propertynames(table))) columns and $(length(table)) rows")
Base.show(io::IO,sys::PanelSystem) = print(io,"PanelSystem($(length.(sys.boundaries)) panels, $(sys.domain))")
function Base.show(io::IO,::MIME"text/plain",sys::PanelSystem)
    println(io,"PanelSystem")
    println(io,"  domain: ",sys.domain)
    println(io,"  background flow U: ",sys.U)
    foreach(b->(print(io,"  boundary: "); show(io,b)),sys.boundaries)
    print(io,"  body area & volume: ",bodyarea(sys),", ",bodyvol(sys))
end

"""
    Φ(x,sys)

Potential `Φ(x) = ∫ₛ q(x')ϕ(x-x')da'` induced by **solved** panel system `sys`.
"""
Φ(x,sys) = sum(map(b->Φ_boundary(x,b,sys.domain),sys.boundaries))
@inline Φ_boundary(x,panels,d) = (G = green(d); sum(m->(xₘ = x .* m; sum(p->p.q*G(xₘ,p),panels)),d.mirrors))
Φₙ(p,sys) = derivative(t->Φ(p.x+t*p.n,sys),0)
∇Φ(x,sys) = gradient(x′->Φ(x′,sys),x)

# Solid body boundaries: Φₙ = -Uₙ
const Body = Union{Table,PanelTree}
isbody(boundary) = true
bodies(sys) = filter(isbody,sys.boundaries)
@inline set_q!(table::Table,q) = table.q .= q
rhs(b::Body,sys) = map(n->-sys.U'n,b.n)
bc!(r,b::Body,sys) = AK.foreachindex(i->r[i] = Φₙ(b[i],sys),r)
precon!(z,boundary) = z # identity

"""
    u([x::SVector{3},] sys)

Measure the velocity vector `u = U+∇Φ`. If no location `x` is given, a vector of
u at all body centers is calculated and is accelerated when Threads.nthreads()>1.

See also: [`Φ`](@ref)
"""
u(x::SVector{3},sys) = sys.U+∇Φ(x,sys)
u(sys) = reduce(vcat,map(b->u(b,sys),bodies(sys)))
u(b,sys) = mapbody!(u,similar(b.x),b,sys)
mapbody!(f,a,b,sys) = (AK.foreachindex(i-> a[i] = f(b.x[i],sys), a); a)

"""
    cₚ([x::SVector{3},] sys)

Measure the pressure coefficient cₚ = 1-u²/U², where `U` is the background velocity and
`u = U+∇Φ` is the flow velocity. If no location `x` is given, a vector of cₚ at all body
centers is calculated and is accelerated when Threads.nthreads()>1.

See also: [`Φ`](@ref)
"""
cₚ(x::SVector{3},sys) = 1-sum(abs2,u(x,sys))/sum(abs2,sys.U)
cₚ(sys) = reduce(vcat,map(b->cₚ(b,sys),bodies(sys)))
cₚ(b,sys) = mapbody!(cₚ,similar(b.q),b,sys)

"""
    steadyforce(sys; S=bodyarea(sys))

Integrated steady pressure force coefficient vector `∫ₛ cₚ nᵢ da/S = Fᵢ/(½ρU²S)`, where `S` is
the body panel area. Computation is accelerated when Threads.nthreads()>1.

See also: [`cₚ`](@ref)
"""
steadyforce(sys;S=bodyarea(sys)) = -sum(b->surface_integral(cₚ(b,sys),b),bodies(sys))/S
@inline surface_integral(a,b) = sum(i->a[i]*b.n[i]*b.dA[i],eachindex(a))

"""
    addedmass(sys; V=bodyvol(sys))

Added mass coefficient force vector `-∫ₛ Φⱼ/Uⱼ nᵢ da/V = mᵢⱼ/ρV` induced by a panel system,
where `V` is the body volume. Computation is accelerated when Threads.nthreads()>1.

**Note:** The index j is set by the velocity vector used to solve the system. For example,
using `U = [0,-1,0]`, means j=2 and addedmass returns the mᵢ₂ vector. Call this function for
j=1:3 to fill in the full added mass matrix,

See also: [`Φ`](@ref)
"""
addedmass(sys;V=bodyvol(sys)) = -sum(b->surface_integral(mapbody!(Φ,similar(b.q),b,sys),b),bodies(sys))/norm(sys.U)/V
"""
    addedmass(panels::Table)

Convenience function to fill in the full added mass matrix via direct solve.
"""
function addedmass(panels::Table;sys=PanelSystem(panels),V=bodyvol(sys))
    A = influence(sys)
    B = panels.n |> stack # source _matrix_ over i=1,2,3
    Q = A\B'              # solution _matrix_ over i=1,2,3
    map(j->addedmass(set_q!(sys,view(Q,:,j));V),1:3) |>stack
end
bodyarea(sys) = sum(b->sum(b.dA),bodies(sys))
bodyvol(sys) = sum(b->sum(p->p.x'p.n * p.dA,b),bodies(sys)) / 3