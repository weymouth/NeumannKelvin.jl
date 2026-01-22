using FastGaussQuadrature
const xlag,wlag = gausslaguerre(4)
const xg32,wg32 = gausslegendre(32)
"""
    quadgl(f,a=-1,b=1;x,w)

Approximate ∫f(x)dx from x=[a,b] using the Gauss-Legendre weights and points `w,x`.
"""
function quadgl(f;x,w)
    I = 0.
    @simd for i in eachindex(x,w)
        I += w[i]*f(x[i])
    end; I
end
quadgl(f,a,b;x=xg32,w=wg32) = (b-a)/2*quadgl(t->f((b+a)/2+t*(b-a)/2);x,w)

"""
    complex_path(g,dg,rngs;atol=1e-3,γ=one,f=Im(γ*exp(im*g)))

Estimate the integral `∫f(t)dt` from `t=[-∞,∞]` using a complex path, see Gibbs 2024. The 
finite phase ranges `rngs` are integrated along the real line with QuadGK. The range end 
points where `flag=true` are integrated to ±∞ in the complex-plane using `±nsp(t₀,g,dg,γ)`.
"""
@inline function complex_path(g,dg,rngs;γ=one,
    f = t->((u,v)=reim(g(t)); @fastmath γ(t)*exp(-v)*sin(u)))

    # Sum the flagged endpoints and interval contributions
    val = zero(rngs[1][1])
    for i in 1:2:length(rngs)
        (t₁,∞₁),(t₂,∞₂) = rngs[i],rngs[i+1]
        ∞₁ && (val -= nsp(t₁,g,dg,γ))
        val += quadgl(f,t₁,t₂)
        ∞₂ && (val += nsp(t₂,g,dg,γ))
    end; val
end

using Roots
"""
    nsp(h₀,g,dg,γ=one)

Integrate the contributions of `imag(∫γ(h)exp(im*g(h))dh)` from
`h = [h₀,∞]` using numerical stationary phase. The complex path
is found as the roots of `ϵ(h)=g(h)-g(h₀)-im*p=0` where `p` are
Gauss-Laguerre integration points. The pre-function `γ` must be
slowly varying compared to `g` over `h`.
"""
@fastmath function nsp(h₀,g,dg,γ=one;xlag=xlag,wlag=wlag,atol=1e-3)
    # Sum over complex Gauss-Laguerre points
    g₀,h = promote(g(h₀),h₀)
    s = zero(typeof(imag(g₀)))
    for (p,w) in zip(xlag,wlag)
        h = find_zero((h->g(h)-g₀-im*p,dg),h,Roots.Newton();atol)
        s += w*imag(γ(h)*exp(im*g₀)*im/dg(h))
    end;s
end

"""
    finite_ranges(S,g,Δg,R;atol=0.1Δg)

Return pairs of flagged ranges `(a₁,f₁),(a₂,f₂)` covering the points `a∈S∈[-R,R]`
such that `|g(a)-g(aᵢ)|≈Δg`. Ranges do no overlap and limited to `±R`. "Unbounded" flag
`fᵢ=false` if `aᵢ=±R`.
"""
function finite_ranges(S::NTuple{N}, g, Δg, R; atol=Δg/10) where N
    Sv, Rv, gv = map(value,S), value(R), t->value(g(t)) # no Duals
    # helper functions to offset the phase and flag if there's no root
    dg(a) = t->abs(gv(a)-gv(t))-Δg
    no(a,b) = abs(gv(a)-gv(b)) ≤ Δg+atol
    # find roots of dg using brackets (Order0) or secant method (Order1)
    fz0(a,b) = no(a,b) ? (return b, false) : (find_zero(dg(a), (a,b), Order0()), true)
    fz1(a,b) = (isfinite(b) && no(a,b)) ? (return b, false) : (find_zero(dg(a), (a,a+copysign(1,b)), Order1(); atol), true)    
    # return flagged sub-range
    (fz1(first(Sv), -Rv), mid_ranges(Val(N), Sv, fz0)..., fz1(last(Sv), Rv))
end
using TupleTools
mid_ranges(::Val{N}, S, fz) where N = TupleTools.vcat(ntuple(N-1) do i
    a, b = S[i], S[i+1]
    p, q = fz(a, b), fz(b, a)
    p[1] < q[1] && return p, q
    c = ((p[1]+q[1])/2, false)
    return c,c
end...)