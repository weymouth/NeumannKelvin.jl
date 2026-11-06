"""
    NKPanelSystem(body; ℓ, Umag=1, sym_axes=(), filter=true, contour=false)

A PanelSystem which applies the Neumann-Kelvin Green's function on the `body`.

**Note**: The free surface is at `z=0` and the direction of the flow is `Û=[-1,0,0]`.
Translate and rotate the body as needed relative to these reference points.

Keyword arguments:
- `ℓ=Umag²/g` Kelvin length
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
function NKPanelSystem(body; ℓ, Umag=1, sym_axes=(), filter=true, contour=false)
    any(x->x[3]≥0,body.x) && throw(ArgumentError("NK panels must be below z=0"))
    3 ∈ sym_axes && throw(ArgumentError("Cannot reflect NK panels above z=0"))
    NKPanelSystem(Table(body,q=zeros_like(body.dA)), SA[-abs(Umag),0,0], mirrors(sym_axes...), (;ℓ,filter,contour))
end

Base.show(io::IO, sys::NKPanelSystem) = println(io, "NKPanelSystem($(length(sys.body)) panels, ℓ=$(sys.args.ℓ))")
Base.show(io::IO, ::MIME"text/plain", sys::NKPanelSystem) = (println(io,"NKPanelSystem");
    println(io,"  Neumann-Kelvin args: ",sys.args);abstract_show(io,sys))

# Overload with Neumann-Kelvin potential
Φ(x,sys::NKPanelSystem) = sum(m->sum(p->p.q*∫K(x .* m,p,sys.args),sys.body),sys.mirrors)
influence(sys::NKPanelSystem) = influence(sys.body,sys.mirrors,(x,p)->∫K(x,p,sys.args))
@inline function ∫K(x,p,(;ℓ,filter,contour)::A) where A
    # Waterline contour factor ℓ∫n₁dy
    dy = extent(components(p.verts,2)); dl = hypot(extent(components(p.verts,1)),dy)
    c = contour && onwaterline(p) ? 1-ℓ*dy^2/(dl*p.dA) : one(dy)
    c < 0 && @warn "Waterline ℓ∫n₁dy > ∫da" maxlog=2
    # Evaluate potential with kelvin(x,y,z ≤ -dl) to filter unresolved waves
    z_max = filter ? -dl : -0f0
    ∫G(x,p)-∫G(x .* SA[1,1,-1],p)+p.dA*kelvin(x,p.x;ℓ,z_max)*c
end
@inline onwaterline(p) = any(x->x[3]≥-1e-4,p.verts)

"""
    kelvin(ξ,α;ℓ)

Nearfield and Wavelike Green's function `N+W` for a traveling source at position `α` with Kelvin
length `ℓ ≡ U²/g`. The free surface is at z=0, and the flow direction is Û=[-1,0,0]. See Noblesse 1981.
"""
function kelvin(ξ,α;ℓ=1,z_max=-0f0)
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
    R > 0  ? singular(abs(x),abs(y),z)+chebregions[][1](S) : eltype(S)(-4.0)
end

# Singular part -2/π∫Im[(1+ζ+ζ²/2+ζ³/6)log(ζ)]dt using Newman's (1987) Uₘ,Vₘ, eqs 21-31
function singular(x,y,z) # x,y ≥ 0
    R = hypot(x,y,z); ω = complex(-z,y)/(R+x) # ω = ρe^{iα}/(R+x)
    sum(((p,q,r,isV,m,c),)->c*z^p*y^q*x^r*(isV ? V(m,ω) : U(m,R,x,z,ω)),Sterms)
end
V(m,ω) = W(imag,m,0,ω)
U(m,R,x,z,ω) = m==0 ? 2log((R+x)/4) : m==1 ? -2+2z/(R+x) : (oftype(R,Uc[m])-W(real,m,1,ω)-(m-1)U(m-2,R,x,z,ω))/m
W(f,m,e,ω) = ldexp(one(real(ω)),2-m)/m*sum(k->binomial(m,k)*(-1)^k*(m-2k)^e*f(ω^(m-2k)),0:(m-1)÷2)
const Uc = (0.,-1.,0.,1/4,0.,-1/8,0.,5/64)
const Sterms = [(p,q,k-p-q,isodd(q),k+1+p+2j+isodd(q),(-1)^(p+(q+isodd(q))÷2)*binomial(q÷2,j)/
                (factorial(p)*factorial(q)*factorial(k-p-q))) for k in 0:3 for p in 0:k for q in 0:k-p for j in 0:q÷2]

# Transformations to/from spherical (S) & Cartesian (X) coordinates
S2X(S) = ((R,θ,α²)=S; α=√α²; SA[R*sin(θ),R*cos(θ)*sin(α),-R*cos(θ)*cos(α)])
X2S(x,y,z) = (R = hypot(x,y,z); SA[R,min(asin(abs(x)/R),π/2-eps()),z==0 ? (π/2)^2 : atan(abs(y/z))^2])
r2R(S) = SA[10/S[1],S[2],S[3]] # 1/R mapping for outer zone, see Newman 1987

# Create fast Chebychev interpolator for Ngk
using FastChebInterp,QuadGK,SpecialFunctions
const chebregions = Ref{NTuple{4,FastChebInterp.ChebPoly{3,Float64,Float64}}}()
function makecheb(l,u;xfrm=identity,tol=1e-4,n=0)
    lb,ub = SA[l,0,0],SA[u,π/2-eps(),(π/2)^2];
    S = S2X.(xfrm.(chebpoints((16,16,8),lb,ub)))
    D = Array{Float64}(undef,size(S))
    AK.foreachindex(i->D[i]=Ngk(S[i]...;n),D)
    chebinterp(D,lb,ub;tol) # create interpolation function
end

# Computed desingularized ∫Nᵢdt using adaptive Gauss-Konrad quadrature
using Base.MathConstants: γ
function Ngk(x,y,z;n=0)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)+im*eps()
    Ni(t) = imag(expintx(ζ(t))+evalpoly(ζ(t),(1,1,1/2,1/6)[1:n+1])*log(ζ(t))+γ) # desingularized to O(ζⁿ⁺¹logζ)
    2/π*quadgk(Ni,-1,0,1)[1]
end
Ngk(X::SVector{3}) = Ngk(X...)

# Wave-like disturbance
function wavelike(x::T,y::T,z::T;γ=one)::T where T
    (x≥0 || z≤-10) && return zero(T)                 # trivial case
    xv,yv,zv = value.((x,y,z))                       # strip any Duals for ranges
    R,Δg = √(-10/zv-1),typeof(xv).(5.4)              # heuristic angle & phase limits
    S = filter(s->-R<s<R,stationary_points(xv,yv))   # g'=0 points
    f(t) = γ(t)*exp(z*(1+t^2))*sin(g(x,y,t))         # integrand
    length(S)==0 && return quadgl(f,-R,R)            # real-line case
    rngs = finite_ranges(S,t->g(xv,yv,t),Δg,R)       # finite phase ranges
    4complex_path(t->g(x,y,t)-im*z*(1+t^2),          # complex case: complex phase
                  t->dg(x,y,t)-2im*z*t, rngs; f,γ)   # ...and it's derivative
end
g(x,y,t) = (x+y*t)*kₓ(t)                  # phase function
dg(x,y,t) = (x*t+y*(2t^2+1))/kₓ(t)        # it's derivative
kₓ(t) = ⎷(1+t^2)                          # x-wavenumber
⎷(z::Complex) = π/2≤angle(z)≤π ? -√z : √z # move √ branch-cut
⎷(x) = √x

# Return points where dg=0
function stationary_points(x::T,y::T) where T
    abs(y)≤√eps(T) && return (zero(T),)
    diff = x^2-8y^2
    diff≤√eps() ? (-x/4y,) : @. (-x+(-1,1)*√diff)/4y
end
