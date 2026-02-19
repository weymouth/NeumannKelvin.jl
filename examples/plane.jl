using NeumannKelvin,QuadGK,TupleTools
using NeumannKelvin: nearfield, wavelike, kₓ, finite_ranges, complex_path, nsp, g, dg, quadgl
using ForwardDiff: value
using SpecialFunctions: besselj1,besselh
using NeumannKelvin: stationary_points as S₀
k(t) = t*kₓ(t)
γ(t,b) = abs(t)<√eps(abs(t)) ? one(t)/2 : besselj1(b*k(t))/k(t)
γ1(t,b) = besselh(1,1,b*k(t)) / (2*k(t)*exp(im*b*k(t)))
γ2(t,b) = besselh(1,2,b*k(t)) / (2*k(t)*exp(-im*b*k(t)))

"""
∫₂kelvin(x,y,z) = ∫ √(1-y′^2) (W(x,y-y′,z)+N(x,y-y′,z)) dy′

Integral of the kelvin Green's function along the line `y′∈[-1,1]` with an elliptical strength distribution. 

# Implementation details

The near-field contribution `N` is given by the zonal Chebychev polynomial approximation as in Newman 1987 and can be integrated over the segment with a typical quadrature rule. 

The wavelike contribution `W` is more delicate, given by the integral

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-y′^2) W dy′ = 4π H(-x)∫ J₁(t√(1+t²))/t√(1+t²) exp(z (1+t^2)) sin(g(x,y,t)) dt

which is better regularized than the original integral for z≈0⁻, but still requires the use of a complex path to avoid wasting 1000s of function evaluations on the decaying tail oscillations.
"""
function ∫₂kelvin(x,y,z,b=1)
    # Near-field contribution (fixed quadrature - nearfield is smooth)
    N_int = quadgl(y′->√(1-y′^2)*nearfield(x,y-y′,z), -1, 1)
    
    # Wavelike contribution
    W_int = if abs(y) > b-x/√8                     # outside the wake
        π*wavelike(x,y,z,γ=t->γ(t,b))                    # use wavelike with the Bessel function pre-factor
    else                                           # inside the wake
        T = promote_type(typeof(x),typeof(y),typeof(z))
        (x≥0 || z≤-10) && return zero(T)                 # trivial case
        y = abs(y)

        xv,yv,zv = value.((x,y,z))                       # strip any Duals for ranges
        R,Δg = √(-10/zv-1),typeof(xv).(8)                # heuristic angle & phase limits
        S = filter(s->-R<s<R,TupleTools.sort(TupleTools.vcat(
                S₀(xv,yv),S₀(xv,yv-b),S₀(xv,yv+b))))     # g'=0 points

        f(t) = γ(t,b)*exp(z*(1+t^2))*sin(g(x,y,t))       # integrand
        length(S)==0 && return quadgl(f,-R,R)            # real-line case
        
        rngs = finite_ranges(S,t->g(xv,yv,t),Δg,R)       # finite phase ranges
        g₊(t)=g(x,y+b,t)-im*z*(1+t^2); dg₊(t)=dg(x,y+b,t)-2im*z*t; γ₊(t)=γ1(t,b)
        g₋(t)=g(x,y-b,t)-im*z*(1+t^2); dg₋(t)=dg(x,y-b,t)-2im*z*t; γ₋(t)=γ2(t,b)

        val = zero(f(zero(T)))
        for i in 1:2:length(rngs)
            (t₁,∞₁),(t₂,∞₂) = rngs[i],rngs[i+1]
            ∞₁ && (val -= nsp(t₁,g₊,dg₊,γ₊) + nsp(t₁,g₋,dg₋,γ₋))
            val += quadgl(f,t₁,t₂)
            ∞₂ && (val += nsp(t₂,g₊,dg₊,γ₊) + nsp(t₂,g₋,dg₋,γ₋))
        end
        4π*val
    end    
    return N_int + W_int
end

# Real-line integrand (use in quadgk and plotting)
function ∫₂W(x,y,z,t)
    kx = hypot(1,t); ky = t*kx; kz = 1+t^2
    γ(t,1)*exp(z*kz)*sin(x*kx+y*ky)
end

# Brute-force version for comparison
function brute∫₂k(x,y,z)
    N_int = quadgk(y′->√(1-y′^2)*nearfield(x,y-y′,z),-1,1)[1]
    W_int = x ≥ 0 ? 0 : 4π*quadgk(t->∫₂W(x,y,z,t),-Inf,0,Inf)[1]
    return N_int + W_int
end

# Check the Bessel function integral identity is correct for an easy value of z
x,y,z = -1.,0.5,-1.
quadgk_count(t->4π*∫₂W(x,y,z,t),-Inf,0,Inf)
quadgk_count(y′->√(1-y′^2)*wavelike(x,abs(y-y′),z),-1,1)

# Check the two ∫₂k implementations give the same answer and compare timings
check(y,x=-1.,z=-0.) = begin
    kelvin = @timed ∫₂kelvin(x,y,z)
    brute = @timed brute∫₂k(x,y,z)
    println("y = $y: kelvin = $(kelvin.value), brute = $(brute.value), kelvin time = $(kelvin.time) seconds, brute time = $(brute.time) seconds")
    (y=y, kv = kelvin.value, bv = brute.value, kt = kelvin.time, bt = brute.time)
end
Table(check(y) for y in (0.,0.5,1.,2.,4.))

# x,y,z = -pi,0.95,-0.
# derivative(x′->∫₂kelvin(x′,y,z),x)
# derivative(x′->brute∫₂k(x′,y,z),x)

# using Plots
# plot(-5:0.05:5,t->derivative(x->∫₂W(x,y,z,t),x))
# plot(0:5e-3:2,y->derivative(x′->∫₂kelvin(x′,y,z;rtol=1e-5),x))
# plot(0:0.01:8,y->derivative(x′->∫₂kelvin(x′,y,z;rtol=1e-5),-20.))
# contour(-20:0.1:1,-10:0.1:10,(x,y)->∫₂kelvin(x,y,-0.))