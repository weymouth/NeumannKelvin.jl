using NeumannKelvin,QuadGK
using NeumannKelvin: nearfield, wavelike, kₓ
using SpecialFunctions: besselj1
γ(t) = abs(t)<√eps(abs(t)) ? one(t)/2 : besselj1(t*kₓ(t))/(t*kₓ(t))

"""
∫₂kelvin(x,y,z) = ∫ √(1-y′^2) (W(x,y-y′,z)+N(x,y-y′,z)) dy′

Integral of the kelvin Green's function along the line `y′∈[-1,1]` with an elliptical strength distribution. 

# Implementation details

The near-field contribution `N` is given by the zonal Chebychev polynomial approximation as in Newman 1987 and can be integrated over the segment with a typical quadrature rule. 

The wavelike contribution `W` is more delicate, given by the integral

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-y′^2) W dy′ = 4π H(-x)∫ J₁(t√(1+t²))/t√(1+t²) exp(z (1+t^2)) sin(g(x,y,t)) dt

which is better regularized than the original integral for z≈0⁻, but still requires the use of a complex path outside the kelvin wave angle to avoid wasting 1000s of function evaluations on the decaying tail oscillations.
"""
function ∫₂kelvin(x,y,z;rtol=1e-3)
    # Near-field contribution
    N_int = quadgk(y′->√(1-y′^2)*nearfield(x,y-y′,z), -1, 1; rtol)[1]
    
    # Wavelike contribution
    W_int = if x≥0          # Upstream: no wavelike contribution
        0
    elseif abs(y) < 1-x/√8  # Inside wake: use direct integration        
        4π*quadgk(t->∫₂W(x,y,z,t), -Inf, 0, Inf; rtol)[1]
    else                    # Outside wake: use complex_path with γ=J₁(k)/k weighting
        π*wavelike(promote(x,abs(y),z)...;γ)
    end    
    return N_int + W_int
end

# Real-line integrand (use in quadgk and plotting)
function ∫₂W(x,y,z,t)
    kx = hypot(1,t); ky = t*kx; kz = 1+t^2
    γ(t)*exp(z*kz)*sin(x*kx+y*ky)
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
    (y=y, kv = kelvin.value, bv = brute.value, kt = kelvin.time, bt = brute.time)
end
Table(check(y) for y in (0.,0.5,1.,2.,4.))

x,y,z = -pi,0.95,-0.
derivative(x′->∫₂kelvin(x′,y,z),x)
derivative(x′->brute∫₂k(x′,y,z),x)

using Plots
plot(-5:0.05:5,t->derivative(x->∫₂W(x,y,z,t),x))
plot(0:5e-3:2,y->derivative(x′->∫₂kelvin(x′,y,z;rtol=1e-5),x))
contour(-20:0.1:1,-10:0.1:10,(x,y)->∫₂kelvin(x,y,-0.))