using FastGaussQuadrature
const xlag,wlag = gausslaguerre(4)
const xg32,wg32 = gausslegendre(32)
const xg8,wg8 = gausslegendre(8)
"""
    quadgl(f,a=-1,b=1;x,w)

Approximate ∫f(x)dx from x=[a,b] using the Gauss-Legendre weights and points `w,x`.
"""
quadgl(f;x,w) = sum(i->w[i]*f(x[i]),eachindex(x,w))
quadgl(f,a,b;x=xg32,w=wg32) = (b-a)/2*quadgl(t->f((b+a)/2+t*(b-a)/2);x,w)

"""
    quad_duffy(f,apex,corners)

Integrate `f(u)` over the polygon defined by `corners` (CCW) using a polar (Duffy) transform fanned from `apex`.
"""
quad_duffy(f,apex,corners::SVector{N};x=SA_F32[-1/√3,1/√3],w=SA_F32[1,1]) where N = sum(1:N) do i
    function fp(ρ,s)
        b,d = corners[i]-apex, corners[i%N+1]-corners[i]
        e = b+s*d
        J = abs(b[1]*e[2]-b[2]*e[1])
        f(apex+ρ*e)*ρ*J
    end
    quadgl(ρ->quadgl(s->fp((ρ+1)/2,(s+1)/2);x,w);x,w)
end/2

"""
    complex_path(g,dg,rngs;atol=1e-3,γ=one,f=Im(γ*exp(im*g)))

Estimate the integral `∫f(t)dt` from `t=[-∞,∞]` using a complex path, see Gibbs 2024. The
finite phase ranges `rngs` are integrated along the real line with QuadGK. The range end
points where `flag=true` are integrated to ±∞ in the complex-plane using `±nsp(t₀,g,dg,γ)`.
"""
@inline function complex_path(g,dg,rngs;γ=one,
    f = t->((u,v)=reim(g(t)); @fastmath γ(t)*exp(-v)*sin(u)))

    # Sum the flagged endpoints and interval contributions
    val = zero(f(rngs[1][1]))
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
    finite_ranges(S,g,Δg,R;atol=Δg/10)

Return pairs of flagged ranges `(a₁,f₁),(a₂,f₂)` covering the points `a∈S∈[-R,R]`
such that `|g(a)-g(aᵢ)|≈Δg`. Ranges do no overlap and limited to `±R`. "Unbounded" flag
`fᵢ=false` if `aᵢ=±R`.
"""
function finite_ranges(S::NTuple{N}, g, Δg, R; atol=Δg/10) where N
    # helper functions to offset the phase and flag if there's no root
    dg(a) = t->abs(g(a)-g(t))-Δg
    no(a,b) = abs(g(a)-g(b)) ≤ Δg+atol
    # find roots of dg using brackets (Order0) or secant method (Order1)
    fz0(a,b) = no(a,b) ? (return b, false) : (find_zero(dg(a), (a,b), AlefeldPotraShi()), true)
    fz1(a,b) = (isfinite(b) && no(a,b)) ? (return b, false) : (find_zero(dg(a), (a,a+copysign(1,b)), Order1(); atol), true)
    # return flagged sub-range
    (fz1(first(S), -R), mid_ranges(Val(N), S, fz0)..., fz1(last(S), R))
end
using TupleTools
mid_ranges(::Val{N}, S, fz) where N = TupleTools.vcat(ntuple(N-1) do i
    a, b = S[i], S[i+1]
    p, q = fz(a, b), fz(b, a)
    p[1] < q[1] && return p, q
    c = ((p[1]+q[1])/2, false)
    return c,c
end...)