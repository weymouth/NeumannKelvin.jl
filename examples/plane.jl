using NeumannKelvin,QuadGK
using NeumannKelvin: nearfield
using SpecialFunctions: besselj1
"""
∫₂kelvin(x,y,z) = ∫ √(1-y′^2) (W(x,y-y′,z)+N(x,y-y′,z)) dy′

Line integral of the kelvin Green's function along the line `y′∈[-1,1]` with an elliptical strength distribution. 

# Implementation details

The near-field contribution `N` is given by the zonal Chebychev polynomial approximation as in Newman 1987 and can be integrated over the segment with a typical quadrature rule. 

The wavelike contribution `W` is more delicate, given by the integral

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-y′^2) W dy′ = 4π H(-x)∫ J₁(t√(1+t²))/t√(1+t²) exp(z (1+t^2)) sin(g(x,y,t)) dt

which is much better regularized than the original integral, but still requires the use of a complex path to avoid wasting 1000s of function evaluations on the decaying tail oscillations.
"""
function ∫₂kelvin(x,y,z)
    N_int = quadgk(y′->√(1-y′^2)*nearfield(x,y-y′,z),-1,1)[1]
    # W_int = x ≥ 0 ? 0 : 4π*quadgk(t->∫₂W(x,y,z,t),-Inf,Inf)[1]
    return N_int #+ W_int
end
function ∫₂W(x,y,z,t)
    kx = hypot(1,t); ky = t*kx; kz = 1+t^2
    bessel_term = abs(ky) < 1e-8 ? 0.5 : besselj1(ky)/ky
    exp(z*kz) * bessel_term * sin(x*kx+y*ky)
end

using Plots
begin
    trng = -5:0.01:5; x,z = -1,-0
    plt = plot(xlabel="T", ylabel="∫₂W($x,y,$z)")
    for y in [0,0.5,1,2,4,8]
        plot!(trng, t->∫₂W(x,y,z,t), label="y=$y, ∫₂kelvin=$(round(∫₂kelvin(x,y,z), digits=1))")
    end; plt
end