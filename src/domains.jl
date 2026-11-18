abstract type FluidDomain end
ϕ(x,p,d::FluidDomain) = (G = green(d); sum(map(m->G(x .* m,p),d.mirrors))) # potential of unit-strength panel p

"""
    Unbounded(; sym_axes=())

Unbounded fluid domain. Setting `sym_axes` imposes symmetry using the method of images. For
example, `sym_axes=(2,3)` mirrors each contribution across `y=0,z=0`, so only one quarter of a
centered & symmetric geometry needs to be covered in panels.
"""
struct Unbounded{M} <: FluidDomain; mirrors::M; end
Unbounded(;sym_axes=()) = Unbounded(mirrors(sym_axes...))
Base.show(io::IO,d::Unbounded) = print(io,"Unbounded(mirrors=$(d.mirrors))")
green(::Unbounded) = ∫G

@inline function mirrors(axes...)
    M = length(axes)
    ntuple(combo -> SA[ntuple(i -> any(j -> axes[j] == i && ((combo >> (j-1)) & 1) == 1, 1:M) ? -1 : 1, 3)...], 1 << M)
end

"""
    Kelvin(ℓ; sym_axes=(), filter=true, contour=false)

Fluid domain below a linear free surface at `z=0` with Kelvin length `ℓ≡U²/g`, using the
Neumann-Kelvin Green function. The flow must be `U=[-|U|,0,0]` and the panels must be below `z=0`.

Keyword arguments:
- `sym_axes` Optional symmetry axes, see `Unbounded`
- `filter` Optional flag to filter waves shorter than the panel size
- `contour` Optional flag to include the effect of the waterline contour

The Green's function is accelerated using a Chebychev surrogate for `N`, Newman 1987, and a
numerical stationary phase method for `W`, Gibbs 2024. The waterline contour is included as a
factor on panels touching `z=0`, Baar 1988. A `PanelTree` can't accelerate panels in this domain.
"""
struct Kelvin{T,M} <: FluidDomain
    ℓ::T
    filter::Bool
    contour::Bool
    mirrors::M
end
function Kelvin(ℓ;sym_axes=(),filter=true,contour=false)
    3 ∈ sym_axes && throw(ArgumentError("Cannot reflect Kelvin panels above z=0"))
    Kelvin(ℓ,filter,contour,mirrors(sym_axes...))
end
Base.show(io::IO,d::Kelvin) = print(io,"Kelvin(ℓ=$(d.ℓ), mirrors=$(d.mirrors), filter=$(d.filter), contour=$(d.contour))")
function check(::Kelvin,sys)
    checkU(sys)
    any(b->any(x->x[3]≥0,b.x),sys.boundaries) && throw(ArgumentError("Kelvin panels must be below z=0"))
end
checkU((;U)) = U[1]<0 && iszero(U[2]) && iszero(U[3]) || throw(ArgumentError("U must be [-|U|,0,0]"))

green(d::Kelvin) = (x,p)->∫K(x,p,d)
@inline function ∫K(x,p,(;ℓ,filter,contour)::Kelvin)
    # Waterline contour factor ℓ∫n₁dy
    dy = extent(components(p.verts,2)); dl = hypot(extent(components(p.verts,1)),dy)
    c = contour && onwaterline(p) ? 1-ℓ*dy^2/(dl*p.dA) : one(dy)
    c < 0 && @warn "Waterline ℓ∫n₁dy > ∫da" maxlog=2
    # Evaluate potential with kelvin(x,y,z ≤ -dl) to filter unresolved waves
    z_max = filter ? -dl : -zero(dl)
    ∫G(x,p)-∫G(x .* SA[1,1,-1],p)+c*∫kelvin(x,p,p.kernel,ℓ,z_max)
end
@inline onwaterline(p) = any(x->x[3]≥-1e-4,p.verts)
∫kelvin(x,p,::GreenKernel,ℓ,z_max) = p.dA*kelvin(x,p.x;ℓ,z_max)
∫kelvin(x,p,::Dipole,ℓ,z_max) = derivative(t->p.dA*kelvin(x,p.x+t*p.n;ℓ,z_max),0) # kelvin isn't translation invariant in z
function ∫kelvin(x,p,::Wake,ℓ,z_max) # stopgap until #27: half-wave panels where waves reach x, then a sinh-mapped tail
    (v₁,v₂),n = p.verts,p.n; e = v₂-v₁; w = norm(e×SA[1,0,0])/2
    sum((-1/√3,1/√3)) do g
        v = v₁+(1+g)/2*e; tw = max(0,value(v[1]-x[1])); d = max(hypot(value(x[2]-v[2]),value(x[3]+v[3])),ℓ/10)
        f(t) = derivative(τ->kelvin(x,v-SA[t,0,0]+τ*n;ℓ,z_max),0)
        m = ceil(Int,tw/(π*ℓ))
        wave = sum(i->quadgl(f,(i-1)*tw/m,i*tw/m;x=xg16,w=wg16),1:m;init=zero(eltype(x)))
        w*(wave+quadgl(u->d*cosh(u)*f(tw+d*sinh(u)),0,12))
    end
end