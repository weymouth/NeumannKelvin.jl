abstract type AbstractPanelSystem end
"""
    PanelSystem(body; freesurf=nothing, sym_axes=(), ℓ=0)

Represents a panel **system**, i.e., a set of panels with strengths `q` used to
satisfy the boundary conditions for the Green's function `∫G(x,p)`.

The system consists of:
- `body`: Required body panel table
- `freesurf`: Optional free surface panel table and Froude-length `l=U²/g`
- `sym_axes`: Optional symmetry axes, see below

The combined panel table is stored in `sys.panels`, with views to the body and free
surface portions available as `sys.body` and `sys.freesurf`. The strength `q` is
initialized to zero and added as a column to `panels`.

Setting `sym_axes` imposes symmetry conditions on the solution using the method of
images. For example, `sym_axes=(2,3)` mirrors each contribution across `y=0,z=0`, so
only one quarter of a centered & symmetric geometry needs to be covered in panels.

# Usage
```julia
sys = PanelSystem(body_panels)                    # body only
sys = PanelSystem(body_panels, freesurf_panels)   # body + free surface
gmressolve!(sys, atol=1e-6)  # approximate solve - but still O(N²) operations!
extrema(cₚ(sys))       # check solution quality
```
"""
struct PanelSystem{T,B,F,M,L} <: AbstractPanelSystem
    panels::T    # combined body & free surface table with q and fsbc columns
    body::B      # view of body panels
    freesurf::F  # view of free surface panels (or nothing)
    mirrors::M   # mirror contributions
    ℓ::L         # Froude-length
end
function PanelSystem(body; freesurf=nothing, sym_axes=(), ℓ=0)
    panels = add_columns(body, q=zero(eltype(body.dA)), fsbc=false)
    !isnothing(freesurf) && (panels = [panels; add_columns(freesurf, q=zero(eltype(freesurf.dA)), fsbc=true)])
    bview = @view panels[1:length(body)]
    fview = isnothing(freesurf) ? nothing : @view panels[length(body)+1:end]
    PanelSystem(panels, bview, fview, mirrors(sym_axes...), [ℓ])
end
PanelSystem(body,q::AbstractArray;kwargs...) = (sys = PanelSystem(body;kwargs...); sys.body.q .= q; sys)

function add_columns(t::Table; kwargs...)
    col = getproperty(t, first(propertynames(t)))
    new_cols = (name => sim_fill(val, col) for (name, val) in kwargs)
    return Table(t; new_cols...)
end
sim_fill(val,array::AbstractArray) = (a = similar(array,typeof(val)); fill!(a,val); a)
@inline function mirrors(axes...)
    M = length(axes)
    ntuple(combo -> SA[ntuple(i -> any(j -> axes[j] == i && ((combo >> (j-1)) & 1) == 1, 1:M) ? -1 : 1, 3)...], 1 << M)
end

# Pretty printing
Base.show(io::IO, sys::PanelSystem) = print(io, "PanelSystem($(length(sys.panels)) panels")
Base.show(io::IO, ::MIME"text/plain", sys::AbstractPanelSystem) = abstract_show(io,sys)
function abstract_show(io,sys)
    show(io,sys);println()
    println(io, "  body area & volume: $(bodyarea(sys)), $(bodyvol(sys))")
    println(io, "  body panel type: $(eltype(sys.body.kernel))")
    if !isnothing(sys.freesurf)
        println(io, "  free surface panels: $(length(sys.freesurf))")
        println(io, "  Froude length: ℓ=$(ℓ[1])")
    end
    println(io, "  mirrors: $(sys.mirrors)")
    println(io, "  strength extrema: $(extrema(sys.panels.q))")
end
bodyarea(sys) = sum(sys.body.dA)
bodyvol(sys) = sum(p.x'p.n * p.dA for p in sys.body) / 3