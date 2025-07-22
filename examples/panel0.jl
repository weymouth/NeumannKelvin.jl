using NeumannKelvin,HCubature
using NeumannKelvin: nearfield
ϕₙ(z,h) = 2hcubature(xy->nearfield(SA[xy...,z]...),SA[-0.5h,0.],SA[0.5h,0.5h])[1]
ϕₙ(-0.,1e-2)*1e4,-2
derivative(z->ϕₙ(z,1e-2),-0.)*1e2
4hcubature(xy->2/(norm(xy)+abs(xy[1])),SA[0.,0.],SA[0.5,0.5])[1]

using NeumannKelvin,TupleTools,QuadGK,IntervalSets
using NeumannKelvin: Δg_ranges,g,kₓ,∫Wᵢ

function ∫₂wavelike(x,z,a,b,ltol=-5log(10),atol=10exp(ltol))
    (x≥0 || z≤ltol) && return 0.
    # Get ranges for f_a, f_b
    Δg,R = -ltol,min(exp(-0.5ltol),√(ltol/z-1)) # phase width & range limit
    A,B = Δg_ranges.(x,(a,b),Δg,R)              # finite phase ranges

    # Range intersections ∩: use f_m & quadgk
    Δ, m, M = (b-a), (a+b)/2, A ∩ B
    @fastmath f_m(t) = Δ*sinc(Δ*t*kₓ(t)/2π)*sin(g(x,m,t))*exp(z*(1+t^2))
    I_m = 4sum(rng->quadgk(f_m,endpoints(rng)...;atol)[1], M)

    # Range differences \: use f_a, f_b & ∫Wᵢ
    I_m + diff(((a,A),(b,B))) do (y,rngs)
        @fastmath f(t)=-cos(g(x,y,t))/t/kₓ(t)*exp(z*(1+t^2))
        ∫Wᵢ(x,y,z,rngs\M;γ=t->-im/t/kₓ(t),f,atol)
    end
end

# check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ),2ϵ*NeumannKelvin.wavelike(x,0.,z),atol=ϵ^2)
check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ),2ϵ*NeumannKelvin.wavelike(x,0.,z),atol=ϵ^2)
check(-1.,-0.0,1e-3)

∫₂wavelike(-10.,-0.,-1.,0.)
∫₂wavelike(-1.,-0.,-1.,0.)
∫₂wavelike(-0.1,-0.,-1.,0.)

using Plots
plot(range(-30,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5),label="∫Gdy",xlabel="xg/U²")
plot(range(-30,-0,10000),x->derivative(z->∫₂wavelike(x,z,-0.5,0.5),-0.),label="∫dG/dz dy",xlabel="xg/U²")
"""
    ∫wavelike(z, a, b, c, d)

Compute the integral `I = 4∫∫∫Im(exp(i*g(x,y,t))) dxdydt` where `t∈[-∞,∞], x∈[a,b], y∈[c,s]`
and `g(x,y,t)=(x+y*t)√(1+t²)-i*z*(1+t²)`.

## Implementation Details
The x&y-integrals are evaluated analytically to give:

    I = ∑ⱼₖ 4∫Im(fⱼₖ(t)) dt = ∑ⱼₖ 4∫Im(γ(t)*exp(i*g(xⱼ,yₖ,t)))

where `γ(t) = ±1/(t*(1+t²))`. Each integral has a set integration intervals `rngsⱼₖ` and `nsp` 
is used to integrate to ±∞ away from these. To avoid catastrophic cancellation over the interval
intersections `M = ⋂ⱼₖ rngsⱼₖ` (especially near t=0), the combined integrand is used over M:

    I = 4∫f_m dt = 4ΔxΔy∫sinc(Δx√(1+t²)/2π)sinc(Δy*t√(1+t²)/2π)*Im(exp(i*g(mx,my,t))) dt

where `Δᵢ,mᵢ` are the difference and mean of the x,y corners, while `fⱼₖ` is used over `rngsⱼₖ-M`.
"""
function ∫wavelike(z,a,b,c,d;ltol=-5log(10),atol=10exp(ltol),α=0)
    (a≥0 || z≤ltol) && return 0. # Panel downstream or too deep
    b = min(-eps(),b)            # Heaviside limit

    # Get ranges
    x,y = SA[a,b],SA[c,d]        # corners
    Δg,R = -0.5ltol,√(ltol/z-1)  # phase width & range limit
    rngs = Δg_ranges.(x,y',Δg,R) # finite phase ranges
    M = reduce(∩,rngs)           # intersection
    isempty(M) && throw(ArgumentError("∫wavelike: Empty intersection - reduce panel size"))

    # Range intersections: use f_m & quadgk
    Δx, Δy, mx, my = b-a, d-c, (a+b)/2, (c+d)/2
    @fastmath f_m(t) = sinc(Δx*kₓ(t)/2π)*sinc(Δy*t*kₓ(t)/2π)*sin(g(mx,my,t))*exp(z*(1+t^2)-g(0.,α,t)^2)
    all(0 .∉ M) && return Δx*Δy*∫Wᵢ(mx,my,z,M;f=f_m,atol) # can't use γ!
    I_m = 4Δx*Δy*sum(rng->quadgk(f_m,endpoints(rng)...;atol)[1],M)

    # Range differences: use ±γ*f & ∫Wᵢ
    I_m - Δx*Δy*sum(Iterators.product(enumerate(x), enumerate(y))) do ((i,xᵢ), (j,yⱼ))
        @fastmath γ(t) = (-1)^(i+j)/(t*(1+t^2)*Δx*Δy)*exp(-g(0.,α,t)^2)
        @fastmath f(t) = γ(t)*sin(g(xᵢ,yⱼ,t))*exp(z*(1+t^2))
        ∫Wᵢ(xᵢ,yⱼ,z,rngs[i,j]\M; γ,f,atol)
    end
end
ϕₖ(z,h) = ∫wavelike(z,-0.5h,0.5h,-0.5h,0.5h,ltol=-10log(10))
ϕₖ(-0.,1e-3)*1e6,-π/2
w₀(h) = derivative(z->ϕₖ(z,h),-0.)
w₀(1e-3)*1e3,-2

using HCubature
hcubature(xy->NeumannKelvin.wavelike(xy[1],xy[2],-1.),SA[-0.05,-0.05],SA[0.05,0.05])
∫wavelike(-1,-0.05,0.05,-0.05,0.05)
hcubature(xy->NeumannKelvin.wavelike(xy[1],xy[2],-1.),SA[-3√8,-3],SA[-3√8+1,-3+1])
∫wavelike(-1,-3√8,-3√8+1,-3,-3+1)
hcubature(xy->NeumannKelvin.wavelike(xy[1],xy[2],-1.),SA[-33√8,-17],SA[-33√8+1,-17+1])
∫wavelike(-1,-33√8,-33√8+1,-17,-17+1) # two M intervals
hcubature(xy->NeumannKelvin.wavelike(xy[1],xy[2],-1.),SA[-33√8,-33],SA[-33√8+1,-33+1])
∫wavelike(-1,-33√8,-33√8+1,-33,-33+1) # uses fallback

using Plots
plot(range(-30,-0,1000),x->∫wavelike(-0.,x,x+1,-0.5,0.5),label="∫∫Gda",xlabel="xg/U²")
plot(range(-30,-0,1000),x->derivative(x->∫wavelike(-0.,x,x+1,-0.5,0.5),x),label="∫∫dG/dx da",xlabel="xg/U²")
plot(range(-30,-0,1000),x->derivative(z->∫wavelike(z,x,x+1,-0.5,0.5),-0.),label="∫∫dG/dz da",xlabel="xg/U²")
plot!(range(-30,-0,1000),x->derivative(z->∫wavelike(z,x,x+1,-0.5,0.5,α=0.1),-0.),label="∫∫dG/dz da",xlabel="xg/U²")

plot(ylabel="∫∫dG/dz da",xlabel="xg/U²");for α in (0,0.025,0.05,0.1)
    plot!(range(-6,6,1000),y->derivative(z ->∫wavelike(z,-5√8,-5√8+1,y-0.5,y+0.5;α),-0.),label=α)
end;plot!()
plot(ylabel="∫∫dG/dz da",xlabel="xg/U²");for α in (0,0.025,0.05,0.1)
    plot!(range(-12,12,1000),y->derivative(z ->∫wavelike(z,-10√8,-10√8+1,y-0.5,y+0.5;α),-0.),label=α)
end;plot!()

contour(-30:0.5:-0,-15:0.25:15,(x,y)->derivative(z->NeumannKelvin.wavelike(x,y,z),-0.),label="dG/dz",xlabel="xg/U²",levels=-9:2:9,clims=(-10,10))
contour(-30:0.25:-0,-15:0.125:15,(x,y)->derivative(z ->∫wavelike(z,x-0.5,x+0.5,y-0.5,y+0.5),-0.),label="∫∫dG/dz da",xlabel="xg/U²",levels=-9:2:9,clims=(-10,10))
contour(-30:0.25:-0,-15:0.125:15,(x,y)->derivative(z ->∫wavelike(z,x-0.5,x+0.5,y-0.5,y+0.5,α=0.1),-0.),label="∫∫dG/dz da",xlabel="xg/U²",levels=-9:2:9,clims=(-10,10))
contour(-30:0.25:-0,-15:0.125:15,(x,y)->derivative(z ->25∫wavelike(z,x-0.1,x+0.1,y-0.1,y+0.1,α=0.1),-0.),label="∫∫dG/dz da",xlabel="xg/U²",levels=-9:2:9,clims=(-10,10))
