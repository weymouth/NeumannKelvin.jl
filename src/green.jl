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
function kelvin(ξ,α;Fn=1)
    α[3] ≥ 0 && throw(DomainError(α[3],"Source must be below the free surface at ζ=0"))

    # Froude number scaled distances from the source's image
    x,y,z = (ξ[1]-α[1])/Fn^2,abs(ξ[2]-α[2])/Fn^2,(ξ[3]+α[3])/Fn^2

	return source(ξ,α)+(1/hypot(x,y,z)+nearfield(x,y,z)+wavelike(x,y,z))/Fn^2
end

using Base.MathConstants: γ
# Near-field disturbance
function nearfield(x,y,z;xgl=xgl,wgl=wgl)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)
	Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ)
	-2*(1-z/(hypot(x,y,z)+abs(x)))+2/π*quadgl(Ni;xgl,wgl)
end

# Wave-like disturbance
function wavelike(x,y,z;wavemax=4,xgl=xgl,wgl=wgl)
    x≥0 && return 0
    # Set up
    f(t) = exp(z*(1+t^2)); b = √abs(-3log(10)/z)
    ψ(t) = (x+abs(y)*t)*S(1+t^2)
    T₀ = stat_points(x,abs(y),b)

    # Pick an integration method
    waves = sum(abs,ψ.([-b,real.(T₀)...,b]))/2π
    waves<wavemax ? 4b*quadgl(t->f(b*t)*sin(ψ(b*t));xgl,wgl) : 4imag(NSD(T₀,f,ψ))
end
S(z::Complex) = π/2≤angle(z)≤π ? -√z : √z
S(x) = √x
function stat_points(x,y,b)
	y==0 && return [0]
	diff = x^2-8y^2
	diff≤1e-8 && return [(-x+√(diff+0im))/4y]
	return filter(x->abs(x)<b, @. (-x+[-1,1]*√diff)/4y)
end