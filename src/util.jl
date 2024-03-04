using FastGaussQuadrature
xgl32,wgl32 = gausslegendre(32)
xgl128,wgl128 = gausslegendre(128)
"""
    quadgl(f;wgl=[1,1],xgl=[-1/√3,1/√3])

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
@fastmath quadgl(f;wgl=SA[1,1],xgl=SA[-1/√3,1/√3]) = wgl'*f.(xgl)

using SpecialFunctions
using ForwardDiff: derivative, gradient, value, partials, Dual
# Fix automatic differentiation of expintx(Complex(Dual))
# https://discourse.julialang.org/t/add-forwarddiff-rule-to-allow-complex-arguments/108821
function SpecialFunctions.expintx(ϕ::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(ϕ); px, py = partials(x), partials(y)
    z = complex(value(x), value(y)); Ω = expintx(z)
    u, v = reim(Ω); ∂u, ∂v = reim(Ω - inv(z))
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end