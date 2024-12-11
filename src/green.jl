""" source(x,a) 

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/hypot(x-a...)

"""
    kelvin(ξ,α;Fn=1)

Green Function `G(ξ)` for a reflected source at position `α` moving with 
`Fn≡U/√gL`. The free surface is at z=0, the coordinates are scaled by L, 
and the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981.
"""
function kelvin(ξ,α;Fn=1)
    # Check inputs
    α[3] < 0 && throw(DomainError(-α[3],"kelvin: source above z=0"))
    ξ[3] > 0 && throw(DomainError(ξ[3],"kelvin: querying above z=0"))

    # reflected source, nearfield, and wavelike disturbance
    x,y,z = (ξ-α)/Fn^2
    return -source(ξ,α)+(nearfield(x,y,z)+wavelike(x,abs(y),z))/Fn^2
end

# Near-field disturbance via zonal Chebychev polynomial approximation as in Newman 1967 
function nearfield(x::T,y::T,z::T)::T where T
	S = X2S(x,y,z); R = S[1]
	l0 = -2*(1-z/(R+abs(x)))
	R ≥ 10 ? l0+c4(r2R(S)) :
	R ≥ 4  ? l0+c3(S) :
	R ≥ 1  ? l0+c2(S) :
	R > 0  ? l0+c1(S) : -4.0
end

# Brute-force quadrature to create data
using FastChebInterp,QuadGK,SpecialFunctions
using Base.MathConstants: γ
function bruteN(x,y,z;l0=-2*(1-z/(hypot(x,y,z)+abs(x))))
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)+im*eps()
    Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ)
    l0+2/π*quadgk(Ni,-1,0,1)[1]
end
bruteN(X::SVector{3}) = bruteN(X...;l0=0)

# Coordinate transformations
S2X(S) = ((R,θ,α)=S; SA[R*sin(θ),R*cos(θ)*sin(α),-R*cos(θ)*cos(α)])
X2S(x,y,z) = (R = hypot(x,y,z); SA[R,asin(abs(x)/R),atan(abs(y/z))])
r2R(S) = SA[10/S[1],S[2],S[3]] # 1/R mapping for outer zone, see Newman 1967

# Create fast Chebychev polynomials
function makecheb(l,u;map=identity,tol=1e-4)
	lb,ub = SA[l,0,0],SA[u,π/2-eps(),π/2];
	chebinterp(bruteN.(S2X.(map.(chebpoints((16,16,8),lb,ub)))),lb,ub;tol)
end
c1,c2,c3,c4 = makecheb(eps(),1),makecheb(1,4),makecheb(4,10),makecheb(1e-5,1;map=r2R)

# Wave-like disturbance 
function wavelike(x,y,z)
    x≥0 && return 0.
    R = √max(0,-5log(10)/z-1) # radius s.t. f(z,R)=1E-5
    4complex_path(t->g(x,y,t)-im*z*(1+t^2), #complex phase
        t->dg(x,y,t)-2im*z*t,               #it's derivative
        stationary_ranges(x,y,R),t->abs(t)≥R)
end

g(x,y,t) = (x+y*t)*√(1+t^2)           # phase function
dg(x,y,t) = (x*t+y*(2t^2+1))/√(1+t^2) # it's derivative

# Return points where dg=0
function stationary_points(x,y) 
    y==0 && return [0.] 
    diff = x^2-8y^2
    diff≤√eps() ? [-x/4y] : @. (-x+[-1.,1.]*√diff)/4y
end

function stationary_ranges(x,y,R,Δg=3π)
    # Get stationary points and guess radius ρ₀
    S = filter(t->abs(t)<1.1R,stationary_points(x,y))
    ρ₀ = Δg*√(inv(0.5Δg*abs(x)+y^2)+inv(x^2+Δg*abs(y)))

    # Get ranges within R with refined ρ estimate
    rngs = map(enumerate(S)) do (i,t₀)
        ρ = refine_ρ(t₀,t->g(x,y,t),t->dg(x,y,t),ρ₀;s=(-1)^i,Δg)
        @. clamp(t₀-(ρ,-ρ),-R,R)
    end |> x->filter(nonzero,x)
    
    # Merge close ranges
    if length(rngs)>1
        a,b = rngs
        6(b[1]-a[2])<b[2]-a[1] && (rngs = [(a[1],b[2])])
    end
    return rngs
end
