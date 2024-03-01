""" source(x,a) 

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/hypot(x-a...)

"""
    noblesse(ξ,a;Fn=1)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. The free surface is ζ=0, the coordinates are scaled by L and
the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981 for details.
"""
function noblesse(ξ,α;Fn=1)
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
function wavelike(x,y,z; wavemax=4)
    x≥0 && return 0
    # Pick an integration method
    count_waves(x,y,z)<wavemax ? wavegl(x,y,z) : waveNSD(x,y,z)
end
function count_waves(x,y,z)
    ψ(t) = (x+abs(y)*t)*hypot(1,t)
    b = √abs(-3log(10)/z)
    T₀ = stat_points(x,abs(y),b)
    sum(abs,ψ.([-b,real.(T₀)...,b]))/2π
end
function wavegl(x,y,z;xgl=xgl,wgl=wgl)
    b = √(-3log(10)/z)
    Wi(t) = exp(z*(1+t^2))*sin((x+abs(y)*t)*hypot(1,t))
    4b*quadgl(t->Wi(b*t);xgl,wgl)
end
function waveNSD(x,y,z) 
    b = √abs(-3log(10)/z)
	T₀ = stat_points(x,abs(y),b)
    4imag(NSD(T₀,t->exp(z*(1+t^2)),t->(x+abs(y)*t)*S(1+t^2)))
end
S(z::Complex) = π/2≤angle(z)≤π ? -√z : √z
S(x) = √x

function stat_points(x,y,b)
	y==0 && return [0]
	diff = x^2-8y^2
	diff≤1e-8 && return [(-x+√(diff+0im))/4y]
	return filter(x->abs(x)<b, @. (-x+[-1,1]*√diff)/4y)
end

"""
    furth(ξ,a;Fn=1,ltol=-3log(10),kwargs...)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. The free surface is ζ=0, the coordinates are scaled by L and
the apparent velocity direction is Û=[-1,0,0]. See Furth 2016 for details.
Smaller log-tolerance `ltol` will only reduce errors when using a large number of
Gauss-Legendre points. Otherwise, it leads to instability.
"""
function furth(ξ,α;Fn=1,ltol=-3log(10),xgl=xgl,wgl=wgl)
    α[3] ≥ 0 && throw(DomainError(α[3],"Source must be below the free surface at ζ=0"))

    # Froude number scaled distances from the source's image
    x,y,z = (ξ-α .* SA[1,1,-1])/Fn^2

    # Wave-like far-field disturbance
    b = min(Tn(x,y),√max(ltol/z-1,0)); a = max(x/abs(y),-b) # integration limits
    W = ifelse(a≥b || x==y==0, 0., 4*quadgl_ab(T->Wi(x,y,z,T),a,b;xgl,wgl))

    # Near-field disturbance
    T₀ = ifelse(y==0,0,clamp(x/y,-b,b)); S = max(abs(T₀),π) # center & scale
    N = 1/hypot(x,y,z)+2S/π*quadgl_inf(T->Ni(x,y,z,S*T-T₀);xgl,wgl)

    # Total Green function
    return source(ξ,α)+(N+W)/Fn^2
end
Ni(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
Wi(x,y,z,T) = exp((1+T^2)*z-dψ⁴(x,y,T)/dψ⁴(x,y,Tn(x,y)))*sin((x-abs(y)*T)*hypot(1,T))
Tn(x,y) = min(10π/abs(x),√(10π/abs(y)))
dψ⁴(x,y,T) = (x*T-abs(y)*(2T^2+1))^4/(1+T^2)^2