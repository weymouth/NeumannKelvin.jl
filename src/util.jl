using FastGaussQuadrature
xlag,wlag = gausslaguerre(5)
xgl,wgl = gausslegendre(32)
"""
    quadgl(f;wgl=[1,1],xgl=[-1/√3,1/√3])

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
@fastmath quadgl(f;wgl=SA[1,1],xgl=SA[-1/√3,1/√3]) = wgl'*f.(xgl)
quadgl_ab(f,a,b;xgl=xgl,wgl=wgl,h=(b-a)/2,j=(a+b)/2) = h*quadgl(t->f(j+h*t);xgl,wgl)

"""
    complex_path(g,dg,rngs,skp)

Estimate the integral `imag(∫exp(im*g(t))dt)` from `t=[-∞,∞]` using a 
complex path. The ranges of stationary phase `rngs` are integrated 
along the real line using Gauss-Legendre. The range endpoints where 
`skp(t)==false` are integrated in the complex-plane using the phase 
derivative `dg(t)=g'` to find the path of stationary phase.
"""
function complex_path(g,dg,rngs,skp;tol=1e-5)
    length(rngs)==0 && return 0.

    # Compute real-line contributions
    W(a,b) = quadgl_ab(t->imag(exp(im*g(t))),a,b)
    I = if length(rngs)==1 # Split the range @ t=0
        a,b = rngs[1]
        W(a,0)+W(0,b)
    else
        sum(W(rng...) for rng in rngs)
    end

    # Add the end point contributions
    I+sum(enumerate(tuplejoin(rngs...))) do (i,t₀)
        skp(t₀) ? 0. : (-1)^i*nsp(t₀,g,dg)
    end
end
"""
    nsp(h₀,g,dg)

Integrate the contributions of `imag(∫exp(im*g(h))dh)` from 
`h = [h₀,∞]` using numerical stationary phase. The complex path
is found as the roots of `ϵ(h)=g(h)-g(h₀)-im*p=0` where `p`
are Gauss-Laguerre integration points.
"""
@fastmath function nsp(h₀,g,dg;xlag=xlag,wlag=wlag)
    # Sum over complex Gauss-Laguerre points
    s,g₀,h,dϵ = 0.,g(h₀),h₀+0im,dg(h₀)
    for (p,w) in zip(xlag,wlag)
        # Newton step(s) to find h
        ϵ = g(h)-g₀-im*p
        h -= ϵ/dϵ # 1st step
        ϵ,dϵ = g(h)-g₀-im*p,dg(h)
        if abs2(ϵ)>1e-6 # if needed..
            h -= ϵ/dϵ # take 2nd step
            dϵ = dg(h)
        end
        s += w*imag(exp(im*g₀)*im/dϵ)
    end;s
end

"""
Refine radius ρ such that `g(t₀±ρ)-g(t₀) ≈ ±Δg`
"""
@fastmath function refine_ρ(t₀,g,dg,s,ρ,Δg;itmx=3,it=1)
    ϵ = g(t₀)-g(t₀+s*ρ)+s*Δg
    while abs(ϵ)>Δg/2 # doesn't need to be perfect
        ρ += s*ϵ/dg(t₀+s*ρ)
        it+=1; it>itmx && break
        ϵ = g(t₀)-g(t₀+s*ρ)+s*Δg
    end
    return ρ
end

nonzero(t::NTuple{2}) = t[2]>t[1]
@inline tuplejoin(x, y) = (x...,y...)
tuplejoin(t::Tuple) = t

using SpecialFunctions
using ForwardDiff: derivative, gradient, value, partials, Dual
# Fix automatic differentiation of expintx(Complex(Dual))
# https://discourse.julialang.org/t/add-forwarddiff-rule-to-allow-complex-arguments/108821
function SpecialFunctions.expintx(ϕ::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(ϕ); px, py = partials(x), partials(y)
    z = complex(value(x), value(y)); Ω = expintx(z)
    u, v = reim(Ω); ∂u, ∂v = reim(Ω - inv(z))
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end