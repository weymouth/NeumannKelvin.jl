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
# Near-field disturbance via zonal Chebychev approximation as in Newman 1987, with the outer zone fitting N+2/R
function nearfield(x,y,z)
    S = X2S(x,y,z); R = S[1]
    R ≥ 10 && return (10cheb(chebregions[][4],r2R(S))/R-2)/R
    l0 = -2*(1-z/(R+absx(x)))
    R ≥ 4  ? l0+cheb(chebregions[][3],S) :
    R ≥ 1  ? l0+cheb(chebregions[][2],S) :
    R > 0  ? l0+Nlog(absx(x),abs(y),z)+cheb(chebregions[][1],S) : eltype(S)(-4.0)
end
absx(x) = x≥0 ? x : -x # same x=0 side as wavelike's x≥0, so N+W stays smooth
# log(R) part of Newman's singular expansion to O(R⁴logR), the only part not smooth in S
Nlog(x,y,z) = -x*(1+3z/4-x^2/8+y^2/16+5z^2/16)*log(hypot(x,y,z))

# Transformations to/from spherical (S) & Cartesian (X) coordinates
S2X(S) = ((R,θ,α²)=S; α=√α²; SA[R*sin(θ),R*cos(θ)*sin(α),-R*cos(θ)*cos(α)])
X2S(x,y,z) = (R = hypot(x,y,z); SA[R,min(asin(absx(x)/R),π/2-eps()),primal(z)==0 ? (π/2)^2 : atan(abs(y/z))^2])
r2R(S) = SA[max(10/S[1],1e-2),S[2],S[3]] # 10/R mapping for outer zone, held at R=1000 beyond

# Create fast Chebychev interpolator for Ngk
using FastChebInterp,QuadGK,SpecialFunctions
const chebregions = Ref{NTuple{4,FastChebInterp.ChebPoly{3,Float64,Float64}}}()
# Skip the domain check, which rejects Duals on a bound in ForwardDiff ≥1 (JuliaMath/FastChebInterp.jl#42)
cheb(c::FastChebInterp.ChebPoly{N},S) where N = FastChebInterp.evaluate((@. (S-c.lb)*2/(c.ub-c.lb)-1),c.coefs,Val{N}(),1,length(c.coefs))
function makecheb(l,u;xfrm=identity,tol=1e-4,f=Ngk)
    lb,ub = SA[l,0,0],SA[u,π/2-eps(),(π/2)^2];
    S = S2X.(xfrm.(chebpoints((16,16,8),lb,ub)))
    D = Array{Float64}(undef,size(S))
    AK.foreachindex(i->D[i]=f(S[i]...),D)
    chebinterp(D,lb,ub;tol) # create interpolation function
end

# Computed desingularized ∫Nᵢdt using adaptive Gauss-Konrad quadrature
using Base.MathConstants: γ
function Ngk(x,y,z;rtol=√eps())
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)+im*eps()
    Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ) # desingularized, see Newman 1987
    2/π*quadgk(Ni,-1,0,-z/(hypot(y,z)+eps()),1;rtol)[1] # split at Re(ζ)=0 for y ≥ 0
end
function farN(x,y,z)
    R = hypot(x,y,z); x = max(abs(x),1e-6R)
    (R*(Ngk(x,y,z;rtol=1e-12)-2*(1-z/(R+x)))+2)*R/10
end
Ngk(X::SVector{3}) = Ngk(X...)

# Wave-like disturbance
function wavelike(x::T,y::T,z::T;γ=one)::T where T
    (x≥0 || z≤-10) && return zero(T)                 # trivial case
    xv,yv,zv = value.((x,y,z))                       # strip any Duals for ranges
    R,Δg = √(-10/zv-1),typeof(xv).(5.4)              # heuristic angle & phase limits
    S = filter(s->-R<s<R,stationary_points(xv,yv))   # g'=0 points
    f(t) = γ(t)*exp(z*(1+t^2))*sin(g(x,y,t))         # integrand
    length(S)==0 && return 4quadgl0(f,-R,R)          # real-line case
    W(Δg) = 4complex_path(t->g(x,y,t)-im*z*(1+t^2),  # complex case: complex phase
                  t->dg(x,y,t)-2im*z*t,              # ...and it's derivative
                  finite_ranges(S,t->g(xv,yv,t),Δg,R); f,γ) # finite phase ranges
    w = W(Δg); isnan(w) ? W(2Δg/3) : w
end
g(x,y,t) = (x+y*t)*kₓ(t)                  # phase function
dg(x,y,t) = (x*t+y*(2t^2+1))/kₓ(t)        # it's derivative
kₓ(t) = ⎷(1+t^2)                          # x-wavenumber
⎷(z::Complex) = real(z)≤0 && !signbit(imag(z)) ? -√z : √z # move √ branch-cut
⎷(x) = √x

# Return points where dg=0
function stationary_points(x::T,y::T) where T
    abs(y)≤√eps(T) && return (zero(T),)
    diff = x^2-8y^2
    diff≤√eps() ? (-x/4y,) : @. (-x+(-1,1)*√diff)/4y
end
