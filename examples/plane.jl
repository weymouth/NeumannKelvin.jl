using NeumannKelvin,QuadGK
using NeumannKelvin: nearfield, g, dg, ⎷, stationary_points, finite_ranges, complex_path,quadgl
using SpecialFunctions: besselj1
using ForwardDiff: value

"""
∫₂kelvin(x,y,z) = ∫ √(1-y′^2) (W(x,y-y′,z)+N(x,y-y′,z)) dy′

Line integral of the kelvin Green's function along the line `y′∈[-1,1]` with an elliptical strength distribution. 

# Implementation details

The near-field contribution `N` is given by the zonal Chebychev polynomial approximation as in Newman 1987 and can be integrated over the segment with a typical quadrature rule. 

The wavelike contribution `W` is more delicate, given by the integral

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-y′^2) W dy′ = 4π H(-x)∫ J₁(t√(1+t²))/t√(1+t²) exp(z (1+t^2)) sin(g(x,y,t)) dt

which is much better regularized than the original integral, but still requires the use of a complex path outside the kelvin wave angle to avoid wasting 1000s of function evaluations on the decaying tail oscillations.
"""
function ∫₂kelvin(x::T,y::T,z::T) where T
    # Near-field contribution
    N_int = quadgk(y′->√(1-y′^2)*nearfield(x,y-y′,z),-1,1)[1]
    
    # Wavelike contribution: Hybrid approach
    # Use quadgk inside wake (where Bessel oscillates rapidly)
    # Use complex_path outside wake (where Bessel is slowly-varying)
    (x≥0 || z≤-10) && return N_int
    xv,yv,zv = value.((x,y,z))
    
    # Wake width threshold: y ± 1 - x/√8
    wake_width = 1 - xv/√8
    
    W_int = if abs(yv) < wake_width
        # Inside wake: use direct integration
        4π * quadgk(t->∫₂W(x,y,z,t), -Inf, 0, Inf, rtol=1e-3)[1]
    else
        # Outside wake: use complex_path
        R = isfinite(-10/zv-1) ? √(-10/zv-1) : typeof(xv)(10)
        Δg = typeof(xv)(5.4)
        S = filter(s->-R<s<R,stationary_points(xv,yv))
        
        # J₁(k)/k weighting
        function γ(t) 
            ky = t*⎷(1+t^2)
            besselj1(ky)/ky
        end
        
        if length(S)==0
            4π * quadgl(t->γ(t)*exp(z*(1+t^2))*sin(g(x,y,t)), -R, R)
        else
            rngs = finite_ranges(S, t->g(xv,yv,t), Δg, R)
            4π * complex_path(t->g(x,y,t)-im*z*(1+t^2),
                              t->dg(x,y,t)-2im*z*t, rngs; γ)
        end
    end
    
    return N_int + W_int
end

function ∫₂W(x,y,z,t)
    kx = hypot(1,t); ky = t*kx; kz = 1+t^2
    exp(z*kz) * besselj1(ky)/ky * sin(x*kx+y*ky)
end

# Old slow version for comparison (using quadgk on infinite domain)
function ∫₂kelvin_slow(x,y,z)
    N_int = quadgk(y′->√(1-y′^2)*nearfield(x,y-y′,z),-1,1)[1]
    W_int = x ≥ 0 ? 0 : 4π*quadgk(t->∫₂W(x,y,z,t),-Inf,0,Inf,atol=1e-6)[1]
    return N_int + W_int
end

check(y,x=-1.,z=-0.) = begin
    println("Testing y = $y...")
    c_p = @timed ∫₂kelvin(x,y,z)
    qgk = @timed ∫₂kelvin_slow(x,y,z)
    println("  Fast: $(c_p.value) in $(c_p.time)s")
    println("  Slow: $(qgk.value) in $(qgk.time)s")
    (y=y, c_pv = c_p.value, qgkv = qgk.value, c_pt = c_p.time, qgkt = qgk.time)
end
Table(check(y) for y in (0.,0.5,1.,2.,4.))