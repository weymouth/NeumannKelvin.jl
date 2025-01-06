using FastGaussQuadrature
const xlag,wlag = gausslaguerre(5)
const xgl20,wgl20 = gausslegendre(20)
const xgl8,wgl8 = gausslegendre(8)
const xgl2,wgl2 = gausslegendre(2)
"""
    quadgl(f;x,w)

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
function quadgl(f;x,w) 
    I = 0.
    @simd for i in eachindex(x,w)
        I += w[i]*f(x[i])
    end; I
end
quadgl_ab(f,a,b;x=xgl20,w=wgl20) = (b-a)/2*quadgl(t->f((b+a)/2+t*(b-a)/2);x,w)

"""
    complex_path(g,dg,rngs,skp)

Estimate the integral `imag(∫exp(im*g(t))dt)` from `t=[-∞,∞]` using a 
complex path. The ranges of stationary phase `rngs` are integrated 
along the real line using Gauss-Legendre. The range endpoints where 
`skp(t)==false` are integrated in the complex-plane using the phase 
derivative `dg(t)=g'` to find the path of stationary phase.
"""
function complex_path(g,dg,rngs,skp=t->false)
    length(rngs)==0 && return 0.

    # Compute real-line contributions
    @fastmath @inline function f(t)
        u,v = reim(g(t))
        exp(-v)*sin(u)
    end
    I = if length(rngs)==1 # Split the range @ t=0
        a,b = rngs[1]
        quadgl_ab(f,a,0.)+quadgl_ab(f,0.,b)
    else
        sum(quadgl_ab(f,rng...) for rng in rngs)
    end

    # Add the end point contributions
    for rng in rngs, (i,t₀) in enumerate(rng)
        !skp(t₀) && (I+= (-1)^i*nsp(t₀,g,dg))
    end; I
end
"""
    nsp(h₀,g,dg)

Integrate the contributions of `imag(∫exp(im*g(h))dh)` from 
`h = [h₀,∞]` using numerical stationary phase. The complex path
is found as the roots of `ϵ(h)=g(h)-g(h₀)-im*p=0` where `p`
are Gauss-Laguerre integration points.
"""
@fastmath function nsp(h₀::T,g,dg;xlag=xlag,wlag=wlag)::T where T
    # Sum over complex Gauss-Laguerre points
    s,g₀,h,dϵ = zero(T),g(h₀),h₀+0im,dg(h₀)
    for (p,w) in zip(xlag,wlag)
        # Newton step(s) to find h
        ϵ = g(h)-g₀-im*p
        h -= ϵ/dϵ # 1st step
        ϵ,dϵ = g(h)-g₀-im*p,dg(h)
        if abs2(ϵ)>1e-8 # if needed..
            h -= ϵ/dϵ # take 2nd step
            dϵ = dg(h)
        end
        s += w*imag(exp(im*g₀)*im/dϵ)
    end;s
end

"""
Refine radius ρ such that `g(t₀±ρ)-g(t₀) ≈ ±Δg`
"""
@fastmath function refine_ρ(t₀,g,dg,ρ;s=1,Δg=3π,itmx=3,it=1,rtol=0.3)
    ϵ = g(t₀)-g(t₀+s*ρ)+s*Δg
    while abs(ϵ)>rtol*Δg
        ρ += s*ϵ/dg(t₀+s*ρ)
        it+=1; it>itmx && break
        ϵ = g(t₀)-g(t₀+s*ρ)+s*Δg
    end
    ρ<0 && throw(DomainError(ρ))
    return ρ
end