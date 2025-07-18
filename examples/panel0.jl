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

using NeumannKelvin,TupleTools,QuadGK,IntervalSets
using NeumannKelvin: Δg_ranges,g,∫Wᵢ,⎷
"""
    ∫₂wavelike(x, z, a, b)

Compute the integral `I = 4∫∫Im(exp(i*g(y,t))) dt dy` where `t∈[-∞,∞], y∈[a,b]` and `g(y,t)=(x+y*t)√(1+t²)-i*z*(1+t²)`.

## Implementation Details
The integration order is swapped and the y-integral is evaluated analytically to give:

    I = 4∫f_b dt - 4∫f_a dt = 4∫Im(exp(i*g(b,t))/ik(t)) dt - 4∫Im(exp(i*g(a,t))/ik(t))

where `k(t) = t√(1+t²)`. Each integral has a set of `A,B=finite_ranges(a,b)` and `nsp` can be
used to integrate to ±∞ away from these. To avoid catastrophic cancellation over the interval 
intersections M = A ∩ B (especially near t=0), the combined integrand is used over these ranges:

    I = 4∫f_m dt = 4∫Δ*sinc(Δ*k(t)/2π)*Im(exp(i*g(m,t))) dt

where `Δ,m = (b-a),(a+b)/2`, while `f_a,f_b` are evalauted over `A-M,B-M`. 
"""
function ∫₂wavelike(x,z,a,b,ltol=-5log(10),atol=10exp(ltol))
    (x≥0 || z≤ltol) && return 0.
    # Get ranges for f_a, f_b
    Δg,R = -ltol,min(exp(-0.5ltol),√(ltol/z-1)) # phase width & range limit
    A,B = Δg_ranges.(x,(a,b),Δg,R)              # finite phase ranges

    # Range intersections ∩: use f_m & quadgk
    Δ, m, M = (b-a), (a+b)/2, A ∩ B
    @fastmath f_m(t) = Δ*sinc(Δ*k(t)/2π)*sin(g(x,m,t))*exp(z*(1+t^2))
    I_m = 4sum(r->quadgk(f_m,endpoints(r)...;atol)[1], M)
    
    # Range differences \: use f_a, f_b & ∫Wᵢ
    I_m + diff(((a,A),(b,B))) do (y,rngs)
        @fastmath f(t)=-cos(g(x,y,t))/k(t)*exp(z*(1+t^2))
        ∫Wᵢ(x,y,z,rngs\M;γ=t->-im/k(t),f,atol)
    end
end
k(t) = t*⎷(1+t^2)

# check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ),2ϵ*NeumannKelvin.wavelike(x,0.,z),atol=ϵ^2)
check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ),2ϵ*NeumannKelvin.wavelike(x,0.,z),atol=ϵ^2)
check(-1.,-0.0,1e-3)

∫₂wavelike(-10.,-0.,-1.,0.)
∫₂wavelike(-1.,-0.,-1.,0.)
∫₂wavelike(-0.1,-0.,-1.,0.)

using Plots
plot(range(-30,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5),label="∫Gdy",xlabel="xg/U²")
plot(range(-30,-0,10000),x->derivative(z->∫₂wavelike(x,z,-0.5,0.5),-0.),label="∫dG/dz dy",xlabel="xg/U²")
