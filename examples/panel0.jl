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
    ∫wavelike(z, a, b, c, d; α=0)

Compute the integral `I = 4H(-x)∫∫∫Im(exp(i*g(x,y,t))) dxdydt` where `t∈[-∞,∞], x∈[a,b], y∈[c,d]`
`g(x,y,t)=(x+y*t)*√(1+t²)-i*z*(1+t²)` and `H` is the Heaviside function. A regularization term 
`exp(-α²*t²*(1+t²))` can be used to damp out the transverse waves proportional to their wavenumber squared.

## Implementation Details
The x and y-integrals are evaluated analytically to give:

    I = ∑ⱼₖ 4∫Im(fⱼₖ(t)) dt = ∑ⱼₖ 4∫Im(γ(t)*exp(i*g(xⱼ,yₖ,t)))

where `γ(t) = ±1/(t*(1+t²))`. Each corner `(xⱼ<0, yₖ)` defines a range `rngsⱼₖ` over which the phase is finite. 
A combined integrand is used over the intersection `M = ⋂ⱼₖ rngsⱼₖ` to avoid cancellation (especially at `t=0`)

    I = 4∫f_m dt = 4ΔxΔy∫sinc(Δx*√(1+t²)/2π)sinc(Δy*t*√(1+t²)/2π)*Im(exp(i*g(mx,my,t))) dt

where `Δᵢ,mᵢ` are the difference and mean of the corner values. Then `∫Wᵢ(fⱼₖ)` is used over `rngsⱼₖ \\ M`.
"""
function ∫wavelike(z,a,b,c,d;ltol=-5log(10),atol=10exp(ltol),α=0)
    (a≥0 || z≤ltol) && return 0. # Panel downstream or too deep
    b = min(-eps(),b)            # Heaviside limit

    # Get ranges
    x,y = SA[a,b],SA[c,d]        # corners
    Δg,R = -0.5ltol,min(√(√-ltol/α),√(ltol/z-1)) # phase width & range limit
    rngs = Δg_ranges.(x,y',Δg,R,addzero=true) # finite phase ranges
    M = reduce(∩,rngs)           # intersection

    # Range intersections: use f_m & quadgk
    Δx, Δy, mx, my = b-a, d-c, (a+b)/2, (c+d)/2
    @fastmath f_m(t) = sinc(Δx*kₓ(t)/2π)*sinc(Δy*t*kₓ(t)/2π)*sin(g(mx,my,t))*exp(z*(1+t^2)-g(0.,α,t)^2)
    I_m = 4Δx*Δy*sum(rng->quadgk(f_m,endpoints(rng)...;atol)[1],M)

    # Range differences: use ±γ*f & ∫Wᵢ
    I_m - Δx*Δy*sum(Iterators.product(enumerate(x), enumerate(y))) do ((i,xᵢ), (j,yⱼ))
        @fastmath γ(t) = (-1)^(i+j)/(t*(1+t^2)*Δx*Δy)*exp(-α^2*t't*(1+t't))
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

using Plots
plot(ylabel="∫∫dG/dz da",xlabel="yg/U²");for α in (0.025,0.05,0.1,0.2)
    plot!(range(-10,0,1000),y->derivative(z ->∫wavelike(z,-5√8,-5√8+1,y-0.5,y+0.5;α),-0.),label=α)
end;plot!(ylims=(-15,5))

plot(ylabel="∫∫dG/dz da",xlabel="xg/U²");for α in (0.025,0.05,0.1,0.2)
    plot!(range(-20,0,1000),y->derivative(z ->∫wavelike(z,-10√8,-10√8+1,y-0.5,y+0.5;α),-0.),label=α)
end;plot!(ylims=(-10,5))

using NeumannKelvin: quadgl,nearfield
using FastGaussQuadrature,ForwardDiff
xgl,wgl=SVector{10}.(gausslegendre(10))
function ∫surf(ξ,p;ℓ=1,ltol=-5log(10),α=0.05)
    (b,a),(d,c),_ = extrema.(components(p.xᵤᵥ/ℓ))
    ξ = ξ/ℓ
    ϕₙ = quadgl(α->nearfield((ξ-α)...),x=p.x₄/ℓ,w=p.w₄)
    # ϕₙ = hcubature(α->nearfield((ξ-SA[α[1],α[2],-0.])...),SA[b,d],SA[a,c],atol=1e-4)[1]
    ϕₖ = ∫wavelike(min(ξ[3],-0.),ξ[1]-a,ξ[1]-b,ξ[2]-c,ξ[2]-d;ltol,α)
    (ϕₙ+ϕₖ)/ℓ
end

Δg,wg = -0.9:0.2:1, fill(0.2,10)
p = measure_panel((u,v)->SA[u,-v,0],0.,0.,5,5)
unwrap(a) = map(i->a[i],[1,2,4,3])
xp,yp,_ = unwrap.(components(p.xᵤᵥ))
w(x,y) = derivative(z->∫surf(SA[x,y,z],p),-0.)
ζ(x,y) = derivative(x->∫surf(SA[x,y,-0.],p),x)

using Plots
x,y = -25:0.2:5,-10:0.1:10
contourf(x,y,w,aspectratio=:equal,widen=false,
    levels=-29:2:29,clims=(-29,29),
    xlabel="xg/U²",ylabel="yg/U²",
    colorbar_title = "vertical velocity w/q",color = :seismic
);plot!(Shape(xp,yp),c=:grey,alpha=0.5,label="panel")
savefig("examples\\surf_panelζ.png")

contourf(x,y,ζ,aspectratio=:equal,widen=false,
    levels=-27:2:27,clims=(-27,27),
    xlabel="xg/U²",ylabel="yg/U²",
    colorbar_title = "free surface height ζg/qU",color = :seismic,
);plot!(Shape(xp,yp),c=:grey,alpha=0.5,label="panel")
savefig("examples\\surf_panelζ.png")

# Almost!?!
dx = 0.1; x = (-0.5+0.5dx):dx:0.5
p = measure_panel.((u,v)->SA[u,-v,0],x,x',dx,dx) |>Table
A = influence(p,ϕ=∫surf)
heatmap(A,aspectratio=:equal,yflip=true)
q = A\ones(100)
heatmap(reshape(q,(10,10)),aspectratio=:equal)