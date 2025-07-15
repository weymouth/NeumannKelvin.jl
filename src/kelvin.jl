"""
    ∫kelvin(ξ,p;ℓ,d²=4,contour=false,filter=contour)

Integrated disturbance of traveling submerged panel `p` on point `ξ` with Froude length `ℓ ≡ U²/g`.
Uses `∫G` for the source and reflected sink potentials and `kelvin` for the free-surface potential. 
A 2x2 quadrature is used when `|x-p.x|² , (z-p.z)²/ℓ² ≤ d²p.dA`, otherwise it uses the midpoint.
If `contour=true` and `p` touches the `z=0` plane, the contribution from the waterline contour 
`ϕ₀=ℓ∫Gₙₖn₁dy` is included. See Noblesse 1983 and Barr & Price 1988.
If `filter=true`, the `z_max` argument to `kelvin` is used to filter waves too small for the panel.
"""
function ∫kelvin(ξ,p;ℓ=1,d²=4,contour=false,filter=contour)
    p′ = reflect(p,3)          # image panel above z=0
    ϕ = ∫G(ξ,p;d²)-∫G(ξ,p′;d²) # Rankine part
    # Are we far from p′?
    far = (p′.x[3]-ξ[3])^2>d²*p.dA*ℓ^2 && sum(abs2,p′.x-ξ)>d²*p.dA
    # Are we on the waterline?
    dx,dy,_ = extent.(components(p.xᵤᵥ)); dl = hypot(dx,dy)
    c = contour && onwaterline(p) ? 1-ℓ*dy^2/(dl*p.dA) : 1.
    c < 0 && @warn "Waterline ℓ∫n₁dy > ∫da" maxlog=2
    # Integrate with z' ≤ -dl to filter unresolved waves
    z_max = filter ? -dl : -0.
    far && return ϕ+p′.dA*kelvin(ξ,p′.x;ℓ,z_max)*c
    ϕ+quadgl(x->kelvin(ξ,x;ℓ,z_max),x=p′.x₄,w=p′.w₄)*c
end
"""
    kelvin(ξ,α;ℓ)

Green Function `G(ξ)` for a traveling source at reflected position `α` with Froude length `ℓ ≡ U²/g`
excluding the sink term. The free surface is at z=0, and the motion direction is Û=[1,0,0]. See Noblesse 1981.
"""
function kelvin(ξ,α;ℓ=1,z_max=-0.,ltol=-5log(10))
    # Check inputs
    α[3] < 0 && @warn "Source point placed above z=0" maxlog=2
    ξ[3] > 0 && throw(DomainError(ξ[3],"kelvin: querying above z=0"))

    # nearfield, and wavelike disturbance
    x,y,z = (ξ-α)/ℓ; z = min(z,z_max/ℓ)
    return (nearfield(x,y,z)+wavelike(x,y,z,ltol))/ℓ
end
"""
    reflect(x::SVector,axis::Int,project::SVector)

Reflects `x::Svector` across an `axis`. Can supply a multiplier vector `project` instead, returning  `x .* project`.
Scalar `x` are not reflected, and reflect is recursively broadcasted over other structures.

Examples:

    julia> reflect((v=SA[1,1,1],s=12),3)
    (v = [1, 1, -1], s = 12)

---

    reflect(f::Function,arg;op=+)

Creates a symmetric (or _anti_-symmetric, if `op=-`) version of the function `f(x,p;kwargs...)`. For example:

Examples:

    julia> ∫G₂ = reflect(∫G,2);

    julia> p = measure_panel((u,v)->SA[u,v,0],0.5,0.5,1,1);

    julia> ∫G₂(p.x,p) == ∫G(p.x,p)+∫G(p.x,reflect(p,2))
    true
"""
reflect(f::F,arg;op=+) where F<:Function = (x,p;kwargs...)->op(f(x,p;kwargs...),f(x,reflect(p,arg);kwargs...))
reflect(x::SVector{n},axis::Int) where n = SA[ntuple(i->i==axis ? -x[i] : x[i],n)...]
reflect(x::SVector{n},flip::SVector{n}) where n = x.*flip # reflect vectors
reflect(x::Number,arg) = x                                # don't change scalars
reflect(p,arg) = map(q->reflect(q,arg),p)                 # map over everything else
"""
    onwaterline(panel) = any(panel.z≥0)
"""
onwaterline(p) = any(components(p.xᵤᵥ,3) .> -eps())

# Near-field disturbance via zonal Chebychev polynomial approximation as in Newman 1987
function nearfield(x::T,y::T,z::T)::T where T
    if Threads.atomic_xchg!(isfirstcall, false)
        @warn "Creating Chebychev polynomials takes a moment"
        global c1,c2,c3,c4 = makecheb(eps(),1),makecheb(1,4),makecheb(4,10),makecheb(1e-5,1;xfrm=r2R)
    end
    S = X2S(x,y,z); R = S[1]
    l0 = -2*(1-z/(R+abs(x)))
    R ≥ 10 ? l0+c4(r2R(S)) :
    R ≥ 4  ? l0+c3(S) :
    R ≥ 1  ? l0+c2(S) :
    R > 0  ? l0+c1(S) : -4.0
end
isfirstcall = Threads.Atomic{Bool}(true)

# Transformations to/from spherical (S) & Cartesian (X) coordinates
S2X(S) = ((R,θ,α²)=S; α=√α²; SA[R*sin(θ),R*cos(θ)*sin(α),-R*cos(θ)*cos(α)])
X2S(x,y,z) = (R = hypot(x,y,z); SA[R,min(asin(abs(x)/R),π/2-eps()),z==0 ? (π/2)^2 : atan(abs(y/z))^2])
r2R(S) = SA[10/S[1],S[2],S[3]] # 1/R mapping for outer zone, see Newman 1987

# Create fast Chebychev interpolator for Ngk
using FastChebInterp,QuadGK,SpecialFunctions
function makecheb(l,u;xfrm=identity,tol=1e-4)
    lb,ub = SA[l,0,0],SA[u,π/2-eps(),(π/2)^2];
    S = S2X.(xfrm.(chebpoints((16,16,8),lb,ub)))
    D = ThreadsX.map(Ngk,S) # generate data (with multi-threading)
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
function wavelike(x,y,z,ltol=-5log(10),atol=exp(ltol))
    (x≥0 || z≤ltol) && return 0.
    R = √(ltol/z-1)            # radius s.t. log₁₀(f(z,R))=ltol
    S = stationary_points(x,y) # g'=0 points
    rngs = finite_ranges(S,t->g(x,y,t),-0.5ltol,R) # finite phase ranges
    4complex_path(t->g(x,y,t)-im*z*(1+t^2),        # complex phase
                  t->dg(x,y,t)-2im*z*t,rngs;atol)  # it's derivative
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
