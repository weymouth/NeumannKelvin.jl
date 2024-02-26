using FastGaussQuadrature
xgl,wgl = gausslegendre(32)
xgc,wgc = gausschebyshevt(32)
xgH,wgH = gausshermite(4)
"""
    quadgl(f;wgl=[1,1],xgl=[-1/√3,1/√3])

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
@fastmath quadgl(f;wgl=SA[1,1],xgl=SA[-1/√3,1/√3]) = wgl'*f.(xgl)

"""
    quadgl_inf(f;kwargs...)

Approximate ∫f(x)dx from x=[-∞,∞] using `quadgl` with the change of variable x=t/(1-t^2).
"""
@fastmath quadgl_inf(f;kwargs...) = quadgl(t->f(t/(1-t^2))*(1+t^2)/(1-t^2)^2;kwargs...)

"""
    quadgl_ab(f,a,b;kwargs...)

Approximate ∫f(x)dx from x=[a,b] using `quadgl` with the change of variable x=½(a+b+tb-ta).
"""
@fastmath function quadgl_ab(f,a,b;kwargs...)
    h,j = (b-a)/2,(a+b)/2
    h*quadgl(t->f(j+h*t);kwargs...)
end
"""
    NSD(x₀,f,g)

Integrate `∫f(x)exp(im*g(x))dt` from `x=[-∞,∞]` with stationary points `g'(x₀)=0` using 
Numerical Steepest Descent. See Deaño, Huybrechs, and Iserles (2018).
"""
function NSD(x₀,f,g;xgH=xgH,wgH=wgH)
	dg(x) = derivative(g,x); d2g(x) = derivative(dg,x); d3g(x) = derivative(d2g,x)

    # Sum over stationary points
	sum(x₀) do x

        # Find most stable approximation of h' at x
		dh = [Inf,sqrt(2im/d2g(x)),(6im/d3g(x))^(1/3)]
		r = argmin(i->abs2(dh[i]),1:3)

        # Sum over Gauss-Hermite points
		sum(zip(wgH,xgH)) do (w,p)

            # Approximate h(p) as h₀ = x+p*h'
			θ,ρ = angle(p*dh[2]),abs(p*dh[r])
			h₀ = (r==2 || θ>0 || imag(x)>0 ) ? x+ρ*exp(im*θ) : x-ρ*im

            # Find true h(p) to ensure stationary phase
			h = NewtonRaphson(h->g(h)-g(x)-im*p^2,h₀)

            # Evaluate w*f(h)*exp(im*g(h))*h'
			2im*exp(im*g(x))*w*f(h)*p/dg(h)
		end
	end
end

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

using ForwardDiff
# Extend derivative to complex arguments
function ForwardDiff.derivative(f,z::Complex)
    x,y = reim(z)
    ∂x = derivative(x->f(x+im*y),x)
    ∂y = derivative(y->f(x+im*y),y)
    0.5(∂x-im*∂y)
end
"""
    NewtonRaphson(f, x₀, tol=1e-8, itr=0, itmax=20)
    
Find `f(x)=0` using initial guess `x₀` and AutoDiff
"""
function NewtonRaphson(f, x, tol=1e-8, itr=0, itmax=20)
    fx =  f(x)
    while abs2(fx) > tol && itr<itmax
        dx = fx/derivative(f,x)
        x -= dx/(1+abs(dx))
        itr+=1
        fx = f(x)
    end; x
end
