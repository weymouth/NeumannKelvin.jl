using FastGaussQuadrature
const xlag,wlag = gausslaguerre(4)
const xgl,wgl = gausslegendre(20)
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
quadgl(f,a,b;x=xgl,w=wgl) = (b-a)/2*quadgl(t->f((b+a)/2+t*(b-a)/2);x,w)

"""
    complex_path(g,dg,rngs,skp)

Estimate the integral `imag(∫exp(im*g(t))dt)` from `t=[-∞,∞]` using a 
complex path. The ranges of stationary phase `rngs` are integrated 
along the real line using Gauss-Legendre. The range endpoints where 
`skp(t)==false` are integrated in the complex-plane using the phase 
derivative `dg(t)=g'` to find the path of stationary phase.
"""
function complex_path(g,dg,rngs,skp=t->false)
    # Compute real-line contributions
    @fastmath @inline function f(t)
        u,v = reim(g(t))
        exp(-v)*sin(u)
    end
    I = sum(quadgl(f,rng...) for rng in rngs)

    # Add the end point contributions
    for rng in combine(rngs...), (i,t₀) in enumerate(rng)
        !skp(t₀) && (I+= (-1)^i*nsp(t₀,g,dg))
    end; I
end
combine(a,b,c...) = a[2]≥b[1] ? combine((a[1],b[2]),c...) : (a,combine(b,c...)...)
combine(a) = (a,)

using Roots
"""
    nsp(h₀,g,dg)

Integrate the contributions of `imag(∫exp(im*g(h))dh)` from 
`h = [h₀,∞]` using numerical stationary phase. The complex path
is found as the roots of `ϵ(h)=g(h)-g(h₀)-im*p=0` where `p`
are Gauss-Laguerre integration points.
"""
@fastmath function nsp(h₀::T,g,dg;xlag=xlag,wlag=wlag,atol=1e-3)::T where T
    # Sum over complex Gauss-Laguerre points
    s,g₀,h = zero(T),g(h₀),h₀+0im
    for (p,w) in zip(xlag,wlag)
        h = find_zero((h->g(h)-g₀-im*p,dg),h,Roots.Newton();atol)
        s += w*imag(exp(im*g₀)*im/dg(h))
    end;s
end

"""
    finite_ranges(S,g,Δg,R;atol=0.3Δg)

Return a set of ranges `(a₁,a₂)` around each point `a∈S` such that
`|g(a)-g(aᵢ)|≈Δg`. Ranges are limited to `±R` and don't overlap.
"""
@fastmath function finite_ranges(S,g,Δg,R;atol=0.3Δg)
    ga2b(a,b) = abs(g(a)-g(b))
    function fz(a,b)
        isfinite(b) && ga2b(a,b)≤Δg+atol && return b
        find_zero(t->ga2b(a,t)-Δg,(a,a+clamp(b-a,-1,1)),Order1();atol)       
    end
    if length(S)==1 || S[2]>R
        a = first(S)
        (fz(a,-R),a),(a,fz(a,R))
    else
        a,b = S
        (fz(a,-R),fz(a,a/2+b/2)),(fz(b,a/2+b/2),fz(b,R))
    end
end