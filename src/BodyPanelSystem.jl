abstract type AbstractPanelSystem end
"""
    BodyPanelSystem(panels,q=zeros; U = SVector(-1,0,0), sym_axes=(), wrap=identity)

Represents a panel **system**, i.e., a set of `panels` with strengths `q` used to
satisfy the boundary conditions for the Green's function `∫G(x,p)`.

The system consists of:
- `panels::Table`: Panels defining the body geometry. **Note** all panel normals must 
point *into* the fluid region.
- `q::Vector` Option initial strength vector which is added as a column to `sys.body`.
- `U::SVector{3}`: Optional background flow vector.
- `wrap::Function` Optional panel data wrapper, such as PanelTree.
- `sym_axes::Int || Tuple`: Optional symmetry axes, see below.

Setting `sym_axes` imposes symmetry conditions on the solution using the method of
images. For example, `sym_axes=(2,3)` mirrors each contribution across `y=0,z=0`, so
only one quarter of a centered & symmetric geometry needs to be covered in panels.

# Usage
```julia
sys = BodyPanelSystem(body_panels) # body only
gmressolve!(sys, atol=1e-6)        # approximate solve - but still O(N²) operations!
extrema(cₚ(sys))                   # check solution quality
```
"""
struct BodyPanelSystem{P,T,M} <: AbstractPanelSystem
    body::P         # body panels
    U::SVector{3,T} # Background velocity SVector
    mirrors::M      # mirror contributions
end
BodyPanelSystem(body; U = SA[-1,0,0], sym_axes=(), wrap=identity, kwargs...) =
    BodyPanelSystem(snug(body,wrap;kwargs...), U, mirrors(sym_axes...))
@inline snug(table,wrap;kwargs...) = wrap(Table(table; q=zeros_like(table.dA));kwargs...)
@inline zeros_like(array::AbstractArray) = (a = similar(array); a .= 0; a)
@inline function mirrors(axes...)
    M = length(axes)
    ntuple(combo -> SA[ntuple(i -> any(j -> axes[j] == i && ((combo >> (j-1)) & 1) == 1, 1:M) ? -1 : 1, 3)...], 1 << M)
end

# Set/get the strength
BodyPanelSystem(panels,q::AbstractArray;kwargs...) = set_q!(BodyPanelSystem(panels;kwargs...),q)
@inline set_q!(sys,q) = (set_q!(sys.body,q); sys)
@inline set_q!(table::Table,q) = table.q .= q
@inline get_q(sys) = sys.body.q

# Pretty printing
Base.show(io::IO, sys::BodyPanelSystem) = println(io, "BodyPanelSystem($(length(sys.body)) panels")
Base.show(io::IO, body::Table) = println(io,"Table with $(length(propertynames(body))) columns and $(length(body)) rows")
Base.show(io::IO, ::MIME"text/plain", sys::BodyPanelSystem) = (println(io,"BodyPanelSystem"); abstract_show(io,sys))
function abstract_show(io,sys)
    print(  io, "  body: "); show(io,sys.body)
    println(io, "     area & volume: $(bodyarea(sys)), $(bodyvol(sys))")
    println(io, "     panel type: $(eltype(sys.body.kernel))")
    println(io, "  background flow U: $(sys.U)")
    println(io, "  mirrors: $(sys.mirrors)")
    print(io, "  strength extrema: $(extrema(sys.body.q))")
end
bodyarea(sys) = sum(sys.body.dA)
bodyvol(sys) = sum(p->p.x'p.n * p.dA,sys.body) / 3