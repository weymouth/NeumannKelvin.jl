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

Evaluate the integral `∫f(t)dt` for `t ∈ rngs` using a mixed real/complex path. 
`rngs` is a collection of real-line intervals, and are integrated using QuadGK.
An open boundary i.e. `rng = [t₁,t₂)` encodes that the interval should _also_ 
be integrated from the boundary point to ±∞ in the complex-plane using `nsp`. 
i.e. `rng=(-2,1]` is evalauted with `-nsp(-2,g,dg,γ)+quadgk(f,-2,1)`.
"""
function complex_path(g,dg,rngs;atol=1e-3,γ=one,
    f = t->((u,v)=reim(g(t)); @fastmath γ(t)*exp(-v)*sin(u)))

    # Sum the interval contributions
    sum(rngs,init=zero(f(0.))) do rng
        # @show rng
        (t₁,t₂) = endpoints(rng); (∞₁,∞₂) = map(!,closedendpoints(rng))
        ∫f,c,n = quadgk_count(f,t₁,t₂;atol)
        # @show ∫f,c,n
        (∞₁ ? -nsp(t₁,g,dg,γ) : zero(t₁)) + ∫f + (∞₂ ? nsp(t₂,g,dg,γ) : zero(t₂))
    end
end

using Roots
"""
    nsp(h₀,g,dg,γ=one)

Integrate the contributions of `imag(∫γ(h)exp(im*g(h))dh)` from
`h = [h₀,±∞]` using numerical stationary phase. The complex path
satisfies `g(h)=g(h₀)+im*p` where `g` is the complex phase and `p`
are Gauss-Laguerre integration points, and is found using the 
phase derivative `dg=g′(h)` and Newton's method. The amplitude `γ`
must be positive and slowly varying compared to `g` over `h`.
"""
@fastmath function nsp(h₀,g,dg,γ=one;xlag=xlag,wlag=wlag,atol=1e-3)
    # Sum over complex Gauss-Laguerre points
    g₀,h,s = promote(g(h₀),h₀,0.)
    for (p,w) in zip(xlag,wlag)
        h = find_zero((h->g(h)-g₀-im*p,dg),h,Roots.Newton();atol)
        s += w*γ(h)*im/dg(h)
    end; imag(exp(im*g₀)*s)
end

using TupleTools,IntervalSets
import ForwardDiff: value
"""
    finite_ranges(S,g,Δg,R;atol=0.1Δg)

Return open/closed intervals around a set of stationary points `S` based on a phase function `g`.
The closed intervals `Δx=[x₁,x₂]` are defined such that `|g(xᵢ)-g(x)|≈Δg` for `x∈S∈-R..R`. Joint intervals are merged.
The boundaries of Δx are opened to indicate the range continues to ±∞ if they don't touch ±R.
"""
function finite_ranges(S,g,Δg,R;atol=0.1Δg)
    function xᵢ(a,b,check=true)
        !isfinite(b) && return @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,a+copysign(1,b)),Order1();atol)
        check && abs(g(a)-g(b))≤Δg+atol && return b
        @fastmath find_zero(t->abs(g(a)-g(t))-Δg,(a,b),Roots.Brent();atol)
    end

    # Sort the stationary points and handle empty case
    S = filter(s->-R<s<R,TupleTools.sort(S))
    length(S) == 0 && return (-value(R)..value(R),) # no Duals

    # Construct disjoint range enpoints between stationary point pairs
    ends = mapreduce(TupleTools.vcat,zip(Base.front(S),Base.tail(S)),init=()) do (a,b)
        abs(g(a)-g(b)) ≤ 2Δg && return () # skip if insufficient gap
        p,q = xᵢ(a,b,false),xᵢ(b,a,false) # look from left & right
        p < q ? (p,q) : ()                # return if disjoint
    end

    # Add first/last endpoint and create open/closed intervals
    map(Iterators.partition((xᵢ(first(S),-R),ends...,xᵢ(last(S),R)),2)) do (a,b)
        Interval{openif(a>-R),openif(b<R)}(value(a),value(b)) # no Duals
    end
end
openif(flag) = flag ? :open : :closed
Base.:\(a::Interval, b::Interval) = mapreduce(bᶜ-> a ∩ bᶜ, TupleTools.vcat, -b)
Base.:-(a::Interval) = ((-Inf .. a.left),(a.right .. Inf))
Base.:∩(A::Tuple,B::Tuple) = filter(!isempty,Tuple(a ∩ b for a in A, b in B))
Base.:\(A::Tuple,B::Tuple) = mapreduce(a -> foldl(\, B; init = a), TupleTools.vcat, A)