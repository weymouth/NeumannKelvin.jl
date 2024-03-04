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
    z ≥ -0.01 && throw(DomainError(z,"Kelvin scaled vertical distance z>-0.01"))

    # Return source, nearfield, and wavelike disturbance
	return source(ξ,α)+(1/hypot(x,y,z)+nearfield(x,y,z)+wavelike(x,y,z;kwargs...))/Fn^2
end

using Base.MathConstants: γ
# Near-field disturbance
function nearfield(x,y,z;xgl=xgl32,wgl=wgl32)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)
	Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ)
	-2*(1-z/(hypot(x,y,z)+abs(x)))+2/π*quadgl(Ni;xgl,wgl)
end

using QuadGK
# Wave-like disturbance
function wavelike(x,y,z;rtol=1e-4,xgl=xgl128,wgl=wgl128,GK=false)
    x≥0 && return 0.
    (GK || z > -0.025) && return 4/√-z*quadgk(t->Wi(x,y,z,t/√-z),-Inf,Inf;rtol)[1]
    abs(y)<-20z ? wavegl(x,y,z,xgl,wgl) : damped(x,y,z,xgl,wgl)
end
wavegl(x,y,z,xgl,wgl) = (b=√(-3log(10)/z); 4b*quadgl(t->Wi(x,y,z,b*t);xgl,wgl))
function damped(x,y,z,xgl,wgl)
    b = √(10π/abs(y))
    dg4(x,y,t) = (x*t+abs(y)*(2*t^2+1))^4/(1+t^2)^2
    scale = dg4(x,y,-b)+dg4(x,y,b)
    8b*quadgl(t->Wi(x,y,z,2b*t)*exp(-dg4(x,y,t)/scale);xgl,wgl)
end
Wi(x,y,z,t) = exp(z*(1+t^2))*sin((x+abs(y)*t)*hypot(1,t))
