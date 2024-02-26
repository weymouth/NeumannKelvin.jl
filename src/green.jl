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

	return source(ξ,α)+(nearfield(x,y,z)+wavelike(x,y,z))/Fn^2
end

using Base.MathConstants: γ
# Near-field disturbance using Gauss-Chebyshev points
function nearfield(x,y,z;xgc=xgc,wgc=wgc)
    r = hypot(x,y,z)
    ζ(t) = (z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)
	Ni(t) = imag(expintx(ζ(t))+log(ζ(t))+γ)
	f(t) = Ni(t)*√(1-t^2)
	1/r-2*(1-z/(r+abs(x)))+2/π*wgc'*f.(xgc)
end

# Wave-like disturbance using Complex Gauss-Hermite points
function wavelike(x,y,z)
    x≥0 && return 0
	b = √max(-3log(10)/z,0)
	T₀ = stat_points(x,abs(y),b)
	4imag(NSD(T₀,t->exp(z*(1+t^2)),t->(x+abs(y)*t)*sqrt(1+t^2)))
end

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