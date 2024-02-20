""" source(x,a) 

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/hypot(x-a...)

"""
    kelvin(ξ,a;Fn=1,ltol=-3log(10),kwargs...)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. The free surface is ζ=0, the coordinates are scaled by L and
the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981 for details.
Smaller log-tolerance `ltol` will only reduce errors when using a large number of
Gauss-Legendre points. Otherwise, it leads to instability.
"""
function kelvin(ξ,α;Fn=1,ltol=-3log(10),xgl=xgl32,wgl=wgl32)
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