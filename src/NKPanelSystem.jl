"""
    NKPanelSystem(body; ℓ, Umag=1, sym_axes=(), filter=true, contour=false)

A PanelSystem which applies the Neumann-Kelvin Green's function on the `body`.

**Note**: The free surface is at `z=0` and the direction of the flow is `Û=[-1,0,0]`.
Translate and rotate the body as needed relative to these reference points.

Keyword arguments:
- `ℓ=Umag²/g` Froude length
- `Umag` Optional background flow magnitude
- `sym_axes` Optional symmetry axes
- `filter` Optional flag to filter waves shorter than the panel size
- `contour` Optional flag to include the effect of the waterline contour

# Details

The Green's function is accelerated using a Chebychev surrogate for `N`, Newman 1987, and
using a numerical stationary phase method for `W`, Gibbs 2024. The waterline contour integral
`C` can be included as a factor multiplying panels touching `z=0`, Baar 1988, although the
impact is small when the body is thin and destablizing when the body is very blunt. Finally,
wave's smaller than the panel size can be filtered out by enforcing a minimum submergence level.

**Note** You cannot use a PanelTree to accelerate Neumann-Kelvin panels because the Green's
function does not decay uniformly with distance. Therefore a simple relative distance cutoff
θ won't maintain accuracy. Accelerating integrals over Neumann-Kelvin panels is an open problem.
"""
struct NKPanelSystem{B,T,M,A} <: AbstractPanelSystem
    body::B      # body panels
    U::SVector{3,T}
    mirrors::M
    args::A
end
@kwdef mutable struct NKargs{L}
    ℓ::L         # Froude-length
    filter::Bool=true
    contour::Bool=false
end
function NKPanelSystem(body; ℓ, Umag=1, sym_axes=(), kwargs...)
    any(x->x[3]≥0,body.x) && throw(ArgumentError("NK panels must be below z=0"))
    NKPanelSystem(Table(body,q=zeros_like(body.dA)), SA[-abs(Umag),0,0], mirrors(sym_axes...), NKargs(;ℓ,kwargs...))
end
update!(sys::NKPanelSystem;kwargs...) = (update!(sys.args;kwargs...);directsolve!(sys))
update!(a::NKargs;kwargs...) = foreach(((k,v),)->setproperty!(a,k,v), pairs(kwargs))

Base.show(io::IO, sys::NKPanelSystem) = println(io, "NKPanelSystem($(length(sys.body)) panels, ℓ=$(sys.args.ℓ))")
Base.show(io::IO, ::MIME"text/plain", sys::NKPanelSystem) = (println(io,"NKPanelSystem");
    print(io,"  Neumann-Kelvin args: ",sys.args);abstract_show(io,sys))
Base.show(io::IO, a::NKargs) = println(io, "ℓ=$(a.ℓ), filter=$(a.filter), contour=$(a.contour)")

# Overload with Neumann-Kelvin potential
Φ(x,sys::NKPanelSystem) = sum(m->sum(p->p.q*∫NK(x .* m,p,sys.args),sys.body),sys.mirrors)
influence(sys::NKPanelSystem) = influence(sys.body,sys.mirrors,(x,p)->∫NK(x,p,sys.args))
@inline function ∫NK(x,p,(;ℓ,filter,contour)::NKargs)
    # Waterline contour factor ℓ∫n₁dy
    dy = extent(components(p.verts,2)); dl = hypot(extent(components(p.verts,1)),dy)
    c = contour && onwaterline(p) ? 1-ℓ*dy^2/(dl*p.dA) : one(dy)
    c < 0 && @warn "Waterline ℓ∫n₁dy > ∫da" maxlog=2
    # Evaluate potential with kelvin(x,y,z ≤ -dl) to filter unresolved waves
    z_max = filter ? -dl : -0.
    ∫G(x,p)-∫G(x .* SA[1,1,-1],p)+p.dA*kelvin(x,p.x;ℓ,z_max)*c
end
@inline onwaterline(p) = any(x->x[3]≥-1e-4,p.verts)

"""
    kelvin(ξ,α;ℓ)

Nearfield and Wavelike Green Functions `N+W` for a traveling source at position `α` with Froude
length `ℓ ≡ U²/g`. The free surface is at z=0, and the flow direction is Û=[-1,0,0]. See Noblesse 1981.
"""
function kelvin(ξ,α;ℓ=1,z_max=-0.)
    # nearfield, and wavelike disturbance
    x,y,z = (ξ-α .* SA[1,1,-1])/ℓ; z = min(z,z_max/ℓ)
    return (nearfield(x,y,z)+wavelike(x,abs(y),z))/ℓ
end
# Near-field disturbance via zonal Chebychev polynomial approximation as in Newman 1987
function nearfield(x,y,z)
    S = X2S(x,y,z); R = S[1]
    l0 = -2*(1-z/(R+abs(x)))
    R ≥ 10 ? l0+chebregions[][4](r2R(S)) :
    R ≥ 4  ? l0+chebregions[][3](S) :
    R ≥ 1  ? l0+chebregions[][2](S) :
    R > 0  ? l0+chebregions[][1](S) : eltype(S)(-4.0)
end

# Transformations to/from spherical (S) & Cartesian (X) coordinates
S2X(S) = ((R,θ,α²)=S; α=√α²; SA[R*sin(θ),R*cos(θ)*sin(α),-R*cos(θ)*cos(α)])
X2S(x,y,z) = (R = hypot(x,y,z); SA[R,min(asin(abs(x)/R),π/2-eps()),z==0 ? (π/2)^2 : atan(abs(y/z))^2])
r2R(S) = SA[10/S[1],S[2],S[3]] # 1/R mapping for outer zone, see Newman 1987

# Create fast Chebychev interpolator for Ngk
using FastChebInterp,QuadGK,SpecialFunctions
const chebregions = Ref{NTuple{4,FastChebInterp.ChebPoly{3,Float64,Float64}}}()
function makecheb(l,u;xfrm=identity,tol=1e-4)
    lb,ub = SA[l,0,0],SA[u,π/2-eps(),(π/2)^2];
    S = S2X.(xfrm.(chebpoints((16,16,8),lb,ub)))
    D = Array{Float64}(undef,size(S))
    AK.foreachindex(i->D[i]=Ngk(S[i]),D)
    chebinterp(D,lb,ub;tol) # create interpolation function
end

# Computed desingularized ∫Nᵢdt using adaptive Gauss-Konrad quadrature
using Base.MathConstants: γ
function Ngk(x,y,z)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)+im*eps()
    Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ) # desingularized, see Newman 1987
    2/π*quadgk(Ni,-1,0,1)[1]
end
Ngk(X::SVector{3}) = Ngk(X...)

# Wave-like disturbance
function wavelike(x::T,y::T,z::T)::T where T
    (x≥0 || z≤-10) && return zero(T)
    S = stationary_points(x,y)                       # g'=0 points
    rngs = finite_ranges(S,t->g(x,y,t),6,√(-10/z-1)) # finite phase ranges
    4complex_path(t->g(x,y,t)-im*z*(1+t^2),          # complex phase
                  t->dg(x,y,t)-2im*z*t,rngs,         # it's derivative
                  f=t->exp(z*(1+t^2))*sin(g(x,y,t))) # real function
end
g(x,y,t) = (x+y*t)*⎷(1+t^2)               # phase function
dg(x,y,t) = (x*t+y*(2t^2+1))/⎷(1+t^2)     # it's derivative
⎷(z::Complex) = π/2≤angle(z)≤π ? -√z : √z # move √ branch-cut
⎷(x) = √x

# Return points where dg=0
function stationary_points(x,y)
    abs(y)≤√eps() && return (0.,)
    diff = x^2-8y^2
    diff≤√eps() ? (-x/4y,) : @. (-x+(-1,1)*√diff)/4y
end
