using NeumannKelvin
using HCubature
phi(z,h) = 4hcubature(xy->-1/norm(SA[xy...,-z]),SA[0.,0.],SA[0.5h,0.5h])[1]
w(h) = derivative(z->phi(z,h),h^2)
w(0.01)
using Richardson
I,e = extrapolate(w,1.0)
abs(I-2π)≤√e

using NeumannKelvin: nearfield
ϕₙ(z,h) = 2hcubature(xy->nearfield(SA[xy...,-z]...),SA[-0.5h,0.],SA[0.5h,0.5h])[1]
for h = logrange(1,1e-5,6)
    w(z) = derivative(z->ϕₙ(z,h),z)/h
    I,e = extrapolate(w,h)
    @show h,I,e
end

using TupleTools,QuadGK
using NeumannKelvin: stationary_points,finite_ranges,filter,complex_path,g,dg,⎷
"""
    ∫₂wavelike(x, z, a, b)

Compute the integral `I = ∫∫Im(exp(i*g(y,t))) dt dy` where `t∈[-∞,∞], y∈[a,b]` and `g(y,t)=(x+y*t)√(1+t²)-i*z*(1+t²)`.

## Implementation Details
The integration order is swapped to and the y-integral is evaluated analytically in y to give 

    I = I_b-I_a = ∫Im(exp(i*g(b,t))/ik(t)) dt - ∫Im(exp(i*g(a,t))/ik(t))

where `k(t) = t√(1+t²)`. These are integrated using `complex_path` with a few caveats:
Critically, the integrands above are singular at t=0, but there difference is not. 

    I_m = ∫Δ*sinc(Δ*k(t)/2π)*Im(exp(i*g(m,t))) dt

where `Δ,m = (b-a),(a+b)/2`. Therefore, we use `I_a,I_b` for any ranges not covering t=0 
(and for all the nsd tails), and I_m for the union of any ranges covering t=0.
"""
function ∫₂wavelike(x,z,a,b,ltol=-5log(10),atol=exp(ltol))
    (x≥0 || z≤ltol) && return 0.
    # Integrate using g(y), rngs and real integrand f  
    I(y,rngs,f) = 4complex_path(t->g(x,y,t)-im*z*(1+t^2),
                t->dg(x,y,t)-2im*z*t,rngs;γ=t->-im/k(t),f,atol)

    # Map over the end-points
    R = min(exp(-0.5ltol),√(ltol/z-1))
    (Ia,rngsa),(Ib,rngsb) = map((a,b)) do y
        # split ranges based on covering/!covering zero
        S = stationary_points(x,y)
        nzero,czero = split(finite_ranges(S,t->g(x,y,t),-0.5ltol,R))
        # return !covering contribution and covering range
        @fastmath @inline f(t)=-cos(g(x,y,t))/k(t)*exp(z*(1+t^2))        
        I(y,nzero,f),czero
    end

    # Merge the covering rngs and get contribution using g(b)-g(a) ∝ sinc(b-a)
    rngs = extrema(TupleTools.vcat(rngsa,rngsb))
    Δ,m = (b-a),(a+b)/2
    @fastmath @inline Δf(t) = Δ*sinc(Δ*k(t)/2π)*sin(g(x,m,t))*exp(z*(1+t^2))
    return I(b,rngs,Δf)-I(a,rngs,zero)+Ib-Ia
end
k(t) = t*⎷(1+t^2)
coverszero(rng) = ((t₁,t₂)=first.(rng); t₁<0<t₂)
pairs(v::NTuple{N}) where N = ntuple(i->(v[2i-1],v[2i]),N÷2)
shallowflatten(t) = ((x for sub in t for x in sub)...,)
split(rngs::Tuple) = map(f->shallowflatten(filter(f,pairs(rngs))),(!coverszero,coverszero))

check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ),2ϵ*NeumannKelvin.wavelike(x,0.,z),atol=ϵ^2)
check(-1.,-0.01,1e-3)

using Plots
plot(range(-30,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5))
plot(range(-1,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5))
plot(range(-30,-0,1000),x->derivative(z->∫₂wavelike(x,z,-0.5,0.5),-0.))
plot(range(-1,-0,1000),x->derivative(z->∫₂wavelike(x,z,-0.5,0.5),-0.))
