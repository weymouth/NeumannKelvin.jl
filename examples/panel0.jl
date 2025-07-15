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

using NeumannKelvin: stationary_points,finite_ranges,filter,complex_path,g,dg,⎷
function ∫₂wavelike(x,z,a,b,ltol=-5log(10),atol=exp(ltol))
    (x≥0 || z≤ltol) && return 0.
    # Get width, mean and integrand
    Δ,y = b-a,(a+b)/2
    @fastmath @inline f(t) = Δ*sinc(Δ*k(t)/2π)*sin(g(x,y,t))*exp(z*(1+t^2))
    # Get finite ranges
    R = min(exp(-0.5ltol),√(ltol/z-1))               # radius s.t. log₁₀(f(z,R))=ltol
    S = TupleTools.flatten(stationary_points.(x,(a,b,y)))
    rngs = finite_ranges(S,t->g(x,y,t),-ltol,R)      # finite phase ranges
    # Get integral using g(y=b)-g(y=a) for the tails
    I(y;f) = 4complex_path(t->g(x,y,t)-im*z*(1+t^2),
        t->dg(x,y,t)-2im*z*t,rngs,γ=t->-im/k(t);f,atol)
    I(b;f)-I(a,f=zero)
end
k(t) = t*⎷(1+t^2)
check(x,z,ϵ) = isapprox(∫₂wavelike(x,z,-ϵ,ϵ,ltol),2ϵ*NeumannKelvin.wavelike(x,0.,z,ltol),rtol=5ϵ^2)
check(-1.,-0.01,1e-3)

using Plots
plot(range(-50,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5))
plot(range(-0.5,-0,1000),x->∫₂wavelike(x,-0.,-0.5,0.5))
# plot(range(-0.5,-0,1000),x->wavelike(x,-0.01,-0.))
# using FastGaussQuadrature
# xgl,wgl = gausslegendre(100);
# ϕₖ(z,h₁=1,h₂=1) = quadgl(x->∫₂wavelike(-x,-z,-0.5h₂,0.5h₂),0.,0.5h₁;x=xgl,w=wgl)
# for h = logrange(1,1e-5,6)
#     w = derivative(z->ϕₖ(z,h,h),h/10)/h
#     @show h,w
# end
