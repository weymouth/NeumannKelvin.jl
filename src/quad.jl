using FastGaussQuadrature
const xlag,wlag = gausslaguerre(4)
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
quadgl(f,a,b;x=SA[-1/√3,1/√3],w=SA[1,1]) = (b-a)/2*quadgl(t->f((b+a)/2+t*(b-a)/2);x,w)

"""
    complex_path(g,dg,rngs;atol=1e-3,γ=one,f=Im(γ*exp(im*g)))

Estimate the integral `∫f(t)dt` from `t=[-∞,∞]` using a complex path.
The finite phase ranges `rngs` are integrated along the real line
with QuadGK. The range endpoints where `flag=true` are integrated to
±∞ in the complex-plane using `±nsp(t₀,g,dg,γ)`.
"""
function complex_path(g,dg,rngs;atol=1e-3,γ=one,
    f = t->((u,v)=reim(g(t)); @fastmath γ(t)*exp(-v)*sin(u)))

    # Sum the flagged endpoints and interval contributions
    sum(Iterators.partition(rngs,2)) do ((t₁,∞₁),(t₂,∞₂))
        ∫f = f==zero ? f(t₁) : quadgk(f,t₁,t₂;atol)[1]
        (∞₁ ? -nsp(t₁,g,dg,γ) : zero(t₁)) + ∫f + (∞₂ ? nsp(t₂,g,dg,γ) : zero(t₂))
    end
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

using TupleTools
import ForwardDiff: value
value(t::Tuple) = value.(t)
"""
    finite_ranges(S,g,Δg,R;atol=0.1Δg)

Return pairs of flagged ranges `(a₁,f₁),(a₂,f₂)` covering the points `a∈S∈[-R,R]`
such that `|g(a)-g(aᵢ)|≈Δg`. Ranges are disjoint and limited to `±R`. "Unbounded" flag 
`fᵢ=false` if `aᵢ=±R`.
"""
function finite_ranges(S,g,Δg,R;atol=0.1Δg)
    function fz(a,b)
        !isfinite(b) && return @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,a+copysign(1,b)),Order1();atol),true
        abs(g(a)-g(b))≤Δg+atol && return b,false
        @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,b),Roots.Brent();atol),true
    end

    # Sort the stationary points and return special cases
    S = filter(s->-R<s<R,TupleTools.sort(S)); N = length(S)
    N == 0 && return ((-R,false),(R,false)) |> value
    fst,lst = fz(first(S),-R),fz(last(S),R)
    N == 1 && return (fst,lst) |> value

    # Ensure ranges are disjoint and concantenate
    mids = mapreduce(TupleTools.vcat,zip(Base.front(S),Base.tail(S))) do (a,b)
        mid = (a+b)/2               # mid-point
        p,q = fz(a,mid),fz(b,mid)   # looking from left & right
        (p[2] && q[2]) ? (p,q) : () # return if disjoint
    end
    TupleTools.vcat((fst,),mids,(lst,)) |> value # ranges can't be Duals
end