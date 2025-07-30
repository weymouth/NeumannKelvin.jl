using NeumannKelvin,QuadGK,IntervalSets
using NeumannKelvin: g,kₓ,Δg_ranges,∫Wᵢ
function ∫Pwave(a::SVector{2},b::SVector{2},c::SMatrix{M,N};z=-0.,ltol=-5log(10),atol=10exp(ltol)) where {M,N}
    (a₁,a₂),(b₁,b₂) = a,b
    a₁≥0 && return 0.    # Whole interval down-stream
    if b₁≥0              # Split into new interval x=[a₁,-0.]
        ξ₊ = (b₁+a₁)/(a₁-b₁) # locate split in panel coords ξ=[-1,1]
        c = project(c,ξ₊)    # project coefficients onto new interval
        b₁ = -0.; b = SA[b₁,b₂] # new corner point
    end
 
    x,y = SA[a₁,b₁],SA[a₂,b₂]
    rngs = Δg_ranges.(x,y',-0.5ltol,Inf,addzero=true) # finite phase ranges
    rngs₀ = reduce(∩,rngs)           # intersection

    Δx,x₀ = b-a,(a+b)/2
    jm(t) = SVector{M}([im^m*jₘ(m,Δx[1]*kₓ(t)/2) for m in 0:M-1])
    jn(t) = SVector{N}([im^n*jₘ(n,Δx[2]*t*kₓ(t)/2) for n in 0:N-1])
    f₀(t) = imag(transpose(jm(t))*c*jn(t)*exp(im*g(x₀...,t)+z*(1+t^2)))
    I₀ = 4prod(Δx)*sum(rng->quadgk(f₀,endpoints(rng)...;atol)[1],rngs₀)

    Am(s,t) = s*SVector{M}([sum((-1)^j*P′(m,j,s)/(im*Δx[1]*kₓ(t)/2)^(j+1) for j in 0:M-1) for m in 0:M-1])
    An(s,t) = s*SVector{N}([sum((-1)^l*P′(n,l,s)/(im*Δx[2]*t*kₓ(t)/2)^(l+1) for l in 0:N-1) for n in 0:N-1])
    I₀ + prod(Δx)*sum(Iterators.product(1:2,1:2)) do (k₁,k₂)
        γ(t) = transpose(Am((-1)^k₁,t))*c*An((-1)^k₂,t)/4
        ∫Wᵢ(x[k₁],y[k₂],z,rngs[k₁,k₂]\rngs₀;γ,atol)
    end
end
using LegendrePolynomials,StaticArrays
using FastGaussQuadrature
using NeumannKelvin
project(c::SMatrix{M,N},x0::Real) where {M,N} = SMatrix{M,M}([Tx0(m-1,j-1,x0) for m in 1:M, j in 1:M])*c
gl_cache = [gausslegendre(n) for n in 1:15]
NeumannKelvin.quadgl(f;order=length(gl_cache)) = sum(w*f(x) for (x,w) in zip(gl_cache[order]...))
Tx0(m,n,x0) = m>n ? 0.0 : (2m+1)/2*NeumannKelvin.quadgl(u->Pl((u+1)*(x0+1)/2-1,n)*Pl(u,m);order=(n+m)÷2+1)
P′(m,j,s) = j>m ? 0 : s^(m+j)*binomial(m+j,m)*binomial(m,j)*factorial(j)÷2^j
using SpecialFunctions: sphericalbesselj
jₘ(m,x) = m==0 ? sinc(x/π) : sign(x)^m*sphericalbesselj(m,abs(x))

@time ∫Pwave(SA[-5e-4,-5e-4],SA[5e-4,5e-4],SA[1.;;],ltol=-10log(10))*1e6,-pi/2
@time derivative(z->∫Pwave(SA[-5e-4,-5e-4],SA[5e-4,5e-4],SA[1.;;];z,ltol=-10log(10)),-0.)*1e3,-2

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
plot();for x in (0.,-0.5,-1.,-2.)
    plot!(range(-2,2,1000),y->derivative(z->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[1;;];z),-0.),label=x)
end;plot!()
plot();for x in (0.,-0.5,-1.,-2.)
    plot!(range(-2,2,1000),y->derivative(z->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[-1 0 1];z),-0.),label=x)
end;plot!()

plot();for x in (0.,-0.5,-1.,-2.)
    plot!(range(-2,2,1000),y->derivative(y->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[1;;]),y),label=x)
end;plot!()
plot();for x in (0.,-0.5,-1.,-2.)
    plot!(range(-2,2,1000),y->derivative(y->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[-1 0 1]),y),label=x)
end;plot!()

plot();for y in (0.,-0.5,-0.99,-1.01)
    plot!(range(-2,2,1000),x->derivative(z->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[1;;];z),-0.),label=y)
end;plot!()
plot();for y in (0.,-0.5,-0.99,-1.01)
    plot!(range(-2,2,1000),x->derivative(z->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[-2/3 0 2/3];z),-0.),label=y)
end;plot!()

plot(range(-100,-90,1000),x->derivative(y->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[0 1]'),0.01))
plot(range(-100,-90,1000),x->derivative(y->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[-2/3 0 2/3]),0.01))
plot(range(-100,-90,1000),x->derivative(y->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[0 0 0;-2/3 0 2/3]),0.01))

smag(xyz)=norm(gradient(xyz->(NeumannKelvin.wavelike(xyz...)),xyz))
wmag(xyz)=norm(gradient(xyz->((x,y,z)=xyz;∫wavelike(z,x-1,x+1,y-1,y+1)),xyz))
Pmag(xyz)=norm(gradient(xyz->((x,y,z)=xyz;∫Pwave(SA[x-1,y-1],SA[x+1,y+1],SA[1 0 -1];z)),xyz))
plot();for x = -logrange(16,128,4)
    plot!(range(0,1/2,1000),y->√-x*smag(SA[x,y*x,-0.]),label=x)
end;plot!(ylims=(0,150))
plot();for x = -logrange(16,128,4)
    plot!(range(0,1/2,1000),y->√-x*wmag(SA[x,y*x,-0.]),label=x)
end;plot!(ylims=(0,150))
plot();for x = -logrange(16,128,4)
    plot!(range(0,1/2,1000),y->√-x*Pmag(SA[x,y*x,-0.]),label=x)
end;plot!(ylims=(0,150))
plot();for x = -logrange(16,128,4)
    plot!(range(0,0.001,1000),y->√-x*wmag(SA[x,y*x,-0.]),label=x)
end;plot!(ylims=(25,42.5))
plot();for x = -logrange(16,128,4)
    plot!(range(0,0.01,1000),y->√-x*Pmag(SA[x,y*x,-0.]),label=x)
end;plot!(ylims=(25,42.5))
