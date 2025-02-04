using FastGaussQuadrature
const xlag,wlag = gausslaguerre(4)
const xgl,wgl = gausslegendre(16)
const xgl2,wgl2 = gausslegendre(2)
"""
    quadgl(f,a-1,b=1;x,w)

Approximate ∫f(x)dx from x=[a,b] using the Gauss-Legendre weights and points `w,x`.
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

Estimate the integral `imag(∫exp(im*g(t))dt)` from `t=[-∞,∞]` using
a complex path. The finite phase ranges `rngs` are integrated along 
the real line using Gauss-Legendre. The range endpoints where 
`flag=true` are integrated in the complex-plane using the phase 
derivative `dg(t)=g'` and numerical stationary phase.
"""
function complex_path(g,dg,rngs)
    # Make an efficient integrand function for real t
    @fastmath @inline function f(t)
        u,v = reim(g(t))
        exp(-v)*sin(u)
    end

    # Sum the flagged endpoints and interval contributions
    sum(rngs) do ((t₁,f₁),(t₂,f₂))
        -(f₁ ? nsp(t₁,g,dg) : zero(t₁))+quadgl(f,t₁,t₂)+(f₂ ? nsp(t₂,g,dg) : zero(t₂))
    end
end

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

Return a set of flagged ranges `(a₁,f₁),(a₂,f₂)` around each point `a∈S∈[-R,R]` 
such that `|g(a)-g(aᵢ)|≈Δg`. Ranges are limited to `±R` and don't overlap and 
the flag `fᵢ=true` if `aᵢ` is off those limits.
"""
function finite_ranges(S,g,Δg,R;atol=0.3Δg)
    function fz(a,b)
        !isfinite(b) && return @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,a+copysign(1,b)),Order1();atol),true
        abs(g(a)-g(b))≤Δg+atol && return b,false
        @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,b),Roots.Brent();atol),true
    end
    if length(S) == 0
        (((-R,false),(R,false)),)
    elseif length(S) == 1
        a = first(S)
        (fz(a,-R),(a,false)),((a,false),fz(a,R))
    else
        a,b = S
        (fz(a,-R),fz(a,a/2+b/2)),(fz(b,a/2+b/2),fz(b,R))
    end
end