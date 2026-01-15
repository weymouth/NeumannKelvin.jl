abstract type AbstractPanelSystem end
"""
    BodyPanelSystem(panels,q=zeros; U = SVector(-1,0,0), sym_axes=(), wrap=identity)

Represents a panel **system**, i.e., a set of panels with strengths `q` used to
satisfy the boundary conditions for the Green's function `∫G(x,p)`.

The system consists of:
- `panels::Table`: body panels. *Note* all panel normals must point **into** the fluid region.
- `q::Vector` which is added as a column to `panels`.
- `U::SVector{3}`: Optional background flow vector.
- `sym_axes::Int || Tuple`: Optional symmetry axes, see below.
- `wrap::Function` panel data wrapper, such as PanelTree.

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
    panels::P       # body panels
    U::SVector{3,T} # Background velocity SVector
    mirrors::M      # mirror contributions
end
function BodyPanelSystem(body; U = SA[-1,0,0], sym_axes=(), wrap=identity)
    body = Table(body; q=zeros_like(body.dA))
    BodyPanelSystem(wrap(body), U, mirrors(sym_axes...))
end
zeros_like(array::AbstractArray) = (a = similar(array); a .= 0; a)
@inline function mirrors(axes...)
    M = length(axes)
    ntuple(combo -> SA[ntuple(i -> any(j -> axes[j] == i && ((combo >> (j-1)) & 1) == 1, 1:M) ? -1 : 1, 3)...], 1 << M)
end

# Set the strength
BodyPanelSystem(panels,q::AbstractArray;kwargs...) = set_q!(BodyPanelSystem(panels;kwargs...),q)
@inline set_q!(sys::AbstractPanelSystem,q) = (set_q!(sys.panels,q); sys)
@inline set_q!(table::Table,q) = table.q .= q

# Pretty printing
Base.show(io::IO, sys::BodyPanelSystem) = print(io, "BodyPanelSystem($(length(sys.panels)) panels")
Base.show(io::IO, ::MIME"text/plain", sys::AbstractPanelSystem) = abstract_show(io,sys)
function abstract_show(io,sys)
    show(io,sys);println()
    println(io, "  body area & volume: $(bodyarea(sys)), $(bodyvol(sys))")
    println(io, "  body panel type: $(eltype(sys.panels.kernel))")
    println(io, "  background flow: $(sys.U)")
    println(io, "  mirrors: $(sys.mirrors)")
    println(io, "  strength extrema: $(extrema(sys.panels.q))")
end
bodyarea(sys) = sum(sys.panels.dA)
bodyvol(sys) = sum(p->p.x'p.n * p.dA,sys.panels) / 3