""" source(x,a) 

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/hypot(x-a...)

"""
    kelvin(ξ,a;Fn=1)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. The free surface is ζ=0, the coordinates are scaled by L and
the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981 for details.
"""
function kelvin(ξ,α;Fn=1,kwargs...)
    # Froude number scaled distances from the source's image
    image = SA[α[1],α[2],-α[3]]
    x,y,z = (ξ-image)/Fn^2

    # Check inputs
    α[3] ≥ 0 && throw(DomainError(α[3],"Kelvin source above ζ=0"))
    z ≥ 0 && throw(DomainError(z,"Kelvin scaled vertical distance above z=0"))

    # Return source, nearfield, and wavelike disturbance
	return source(ξ,α)+(1/hypot(x,y,z)+nearfield(x,y,z)+wavelike(x,y,z;kwargs...))/Fn^2
end

using Base.MathConstants: γ
# Near-field disturbance
function nearfield(x,y,z;xgl=xgl,wgl=wgl)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)
	Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ)
	-2*(1-z/(hypot(x,y,z)+abs(x)))+2/π*quadgl(Ni;xgl,wgl)
end

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

# Return points where dg=0 as a tuple
function stationary_points(x,y) 
    y==0 && return (0.,) 
    diff = x^2-8y^2
    diff≤√eps() && return (-x/4y,)
    @. (-x+(-1,1)*√diff)/4y
end

function stationary_ranges(x,y,R,Δg=3π)
    # Get stationary points and guess radius ρ₀
    S = filter(t->abs(t)<1.1R,stationary_points(x,y))
    ρ₀ = Δg*√(inv(0.5Δg*abs(x)+y^2)+inv(x^2+Δg*abs(y)))

    # Get ranges within R with refined ρ estimate
    rngs = map(enumerate(S)) do (i,t₀)
        ρ = refine_ρ(t₀,t->g(x,y,t),t->dg(x,y,t),(-1)^i,ρ₀,Δg)
        @. clamp(t₀-(ρ,-ρ),-R,R)
    end |> x->filter(nonzero,x)
    
    # Merge close ranges
    if length(rngs)>1
        a,b = rngs
        6(b[1]-a[2])<b[2]-a[1] && (rngs = ((a[1],b[2]),))
    end
    return rngs
end
