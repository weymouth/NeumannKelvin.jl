using NeumannKelvin,QuadGK,IntervalSets
using NeumannKelvin: g,kₓ,Δg_ranges,∫Wᵢ
using SpecialFunctions: sphericalbesselj
function ∫Pwave(a::SVector{2},b::SVector{2},c::AbstractMatrix;z=-0.,ltol=-5log(10),atol=10exp(ltol))
    (a₁,a₂),(b₁,b₂) = a,b
    a₁≥0 && return 0.    # Whole interval down-stream
    if b₁≥0              # Split into new interval x=[a₁,-eps()]
        ξ₊ = (b₁+a₁)/(a₁-b₁) # locate split in panel coords ξ=[-1,1]
        c = project(c,ξ₊)    # project coefficients onto new interval
        b₁ = -eps(); b = SA[b₁,b₂] # new corner point
    end
 
    x,y = SA[a₁,b₁],SA[a₂,b₂]
    rngs = Δg_ranges.(x,y',-0.5ltol,Inf,addzero=true) # finite phase ranges
    M = reduce(∩,rngs)           # intersection

    Δx,x₀ = b-a,(a+b)/2
    @fastmath j(m,x) = sign(x)^m*sphericalbesselj(m,abs(x))
    @fastmath f₀(t) = imag(exp(im*g(x₀...,t)+z*(1+t^2))*sum(withindex(c)) do (m,n)
        c[m,n]==0 ? 0. : c[m,n]*im^(m+n-2)*j(m-1,Δx[1]*kₓ(t)/2)*j(n-1,Δx[2]*t*kₓ(t)/2)
    end)
    I₀ = 4prod(Δx)*sum(rng->quadgk(f₀,endpoints(rng)...;atol)[1],M)

    @fastmath P′(m,j,s) = j>m ? 0 : (s)^(m+j)*2^j*factorial(m)÷factorial(m-j)
    I₀ + prod(Δx)*sum(withindex(rngs)) do (k₁,k₂)
        @fastmath γ(t) = (-1)^(k₁+k₂)/prod(Δx)*sum(withindex(c)) do (j,l)
            (im*kₓ(t))^(-j)*(im*t*kₓ(t))^(-l)*sum(withindex(c)) do (m,n)
                c[m,n]==0 ? 0. : c[m,n]*P′(m-1,j-1,2k₁-3)*P′(n-1,l-1,2k₂-3)
        end;end
        ∫Wᵢ(x[k₁],y[k₂],z,rngs[k₁,k₂]\M;γ,atol)
    end
end
withindex(A) = (Tuple(I) for I in CartesianIndices(A))
using LegendrePolynomials,StaticArrays
using FastGaussQuadrature
using NeumannKelvin
project(c::SMatrix{M,N},x0::Real) where {M,N} = SMatrix{M,M}([Tx0(m-1,j-1,x0) for m in 1:M, j in 1:M])*c
gl_cache = [gausslegendre(n) for n in 1:15]
NeumannKelvin.quadgl(f;order=length(gl_cache)) = sum(w*f(x) for (x,w) in zip(gl_cache[order]...))
Tx0(m,n,x0) = m>n ? 0.0 : (2m+1)/2*quadgl(u->Pl((u+1)*(x0+1)/2-1,n)*Pl(u,m);order=(n+m)÷2+1)

∫Pwave(SA[-1e-3,-1e-3],SA[1e-3,1e-3],SA[1.;;],ltol=-10log(10))/4e-6,pi/2
derivative(z->∫Pwave(SA[-1e-3,-1e-3],SA[1e-3,1e-3],SA[1.;;];z),-0.)/2e-3,-2

using HCubature
a,b,z = SA[-1.,-1.],SA[1.,1.],-1.
dx,x₀ = b-a,(a+b)/2
hcubature(xy->NeumannKelvin.wavelike(xy[1],xy[2],z),a,b)
∫Pwave(a,b,SA[1.;;];z)
hcubature(a,b) do xy
    x,y= (xy-x₀) ./ 0.5dx # shift to [-1,1]
    x*(y^2-1)*NeumannKelvin.wavelike(xy[1],xy[2],z)
end
∫Pwave(a,b,SA[0 0 0;-2/3 0 2/3];z)

using Plots
c₀,c₁₂,c₂₂=SA[1.;;],SA[0 0 0;-1 0 -2/3],SA[1. 0. -2/3;0 0 0;-2/3 0 4/9]
for c in (c₀,c₁₂,c₂₂)
    @show ∫Pwave(SA[-2.,-2.],SA[-0.,-0.],c)
end
