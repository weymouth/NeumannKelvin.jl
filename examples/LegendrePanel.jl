using NeumannKelvin,QuadGK,IntervalSets
using NeumannKelvin: g,kₓ,Δg_ranges,∫Wᵢ
function ∫Pwave(a::SVector{2},b::SVector{2},c::SMatrix{M,N};z=-0.,α=0,atol=1e-4,ltol=log(atol/10)) where {M,N}
    (a₁,a₂),(b₁,b₂) = a,b
    a₁≥0 && return 0.    # Whole interval down-stream
    if b₁≥0              # Split into new interval x=[a₁,-0]
        ξ₊ = (b₁+a₁)/(a₁-b₁) # locate split in panel coords ξ=[-1,1]
        c = project(c,ξ₊)    # project coefficients onto new interval
        b₁,b = -0,SA[-0,b₂]  # new corner point
    end
 
    x,y = SA[a₁,b₁],SA[a₂,b₂]
    rngs = Δg_ranges.(x,y',-ltol,Inf,addzero=true) # finite phase ranges
    rngs₀ = reduce(∩,rngs)           # intersection

    Δx,x₀ = b-a,(a+b)/2
    jm(t) = SVector{M}([im^m*jₘ(m,Δx[1]*kₓ(t)/2) for m in 0:M-1])
    jn(t) = SVector{N}([im^n*jₘ(n,Δx[2]*t*kₓ(t)/2) for n in 0:N-1])
    f₀(t) = imag(transpose(jm(t))*c*jn(t)*exp(im*g(x₀...,t)+(z-α^2*t^2)*(1+t^2)))
    I₀ = 4prod(Δx)*sum(rng->quadgk(f₀,endpoints(rng)...;atol)[1],rngs₀)

    Am(s,t) = s*SVector{M}([sum((-1)^j*P′(m,j,s)/(im*Δx[1]*kₓ(t)/2)^(j+1) for j in 0:M-1) for m in 0:M-1])
    An(s,t) = s*SVector{N}([sum((-1)^l*P′(n,l,s)/(im*Δx[2]*t*kₓ(t)/2)^(l+1) for l in 0:N-1) for n in 0:N-1])
    I₀ + prod(Δx)*sum(Iterators.product(1:2,1:2)) do (k₁,k₂)
        γ(t) = transpose(Am((-1)^k₁,t))*c*An((-1)^k₂,t)/4*exp(-α^2*t't*(1+t't))
        ∫Wᵢ(x[k₁],y[k₂],z,rngs[k₁,k₂]\rngs₀;γ,atol)
    end
end
using LegendrePolynomials,StaticArrays
using FastGaussQuadrature
project(c::SMatrix{M,N},x0::Real) where {M,N} = SMatrix{M,M}([Tx0(m-1,j-1,x0) for m in 1:M, j in 1:M])*c
gl_cache = [gausslegendre(n) for n in 1:15]
NeumannKelvin.quadgl(f;order=length(gl_cache)) = sum(w*f(x) for (x,w) in zip(gl_cache[order]...))
Tx0(m,n,x0) = m>n ? 0.0 : (2m+1)/2*NeumannKelvin.quadgl(u->Pl((u+1)*(x0+1)/2-1,n)*Pl(u,m);order=(n+m)÷2+1)
P′(m,j,s) = j>m ? 0 : s^(m+j)*binomial(m+j,m)*binomial(m,j)*factorial(j)÷2^j
using SpecialFunctions: sphericalbesselj
jₘ(m,x) = m==0 ? sinc(x/π) : sign(x)^m*sphericalbesselj(m,abs(x))

∫Pwave(SA[-5e-4,-5e-4],SA[5e-4,5e-4],SA[1.;;],atol=1e-8)*1e6,-pi/2
derivative(z->∫Pwave(SA[-5e-4,-5e-4],SA[5e-4,5e-4],SA[1.;;];z,atol=1e-8),-0.)*1e3,-2

using NeumannKelvin,HCubature
using NeumannKelvin:nearfield
function cmat(vec::SVector{N,T}) where {N,T}
    c = zeros(T,2N-1)
    foreach(i->c[2i-1]=vec[i],1:N)
    SMatrix{1,2N-1}(c)
end 
source(x,y,c::SMatrix{M,N}) where{M,N} = [Pl(x,l) for l in 0:M-1]'*c*[Pl(y,l) for l in 0:N-1]
ϕ(y,j) = Pl(y,2j-2)
function ∫Pnear(xyz,c=SA[1;;];atol=1e-4)
    x,y,z = xyz
    if -1<x<1 && -1<y<1 
        # c = ((-1,-1),(1,-1),(1,1),(-1,1))
        # sum(1:4) do i
        #     (x1,y1),(x2,y2) = c[i],c[i%4+1]
        #     dx1,dx2,dy1,dy2 = x1-x, x2-x, y1-y, y2-y
        #     J = abs(dx1*dy2-dx2*dy1)
        #     hcubature(SA[0.,0.],SA[1.,1.];atol) do (s,t)
        #         u,v = s*t,s*(1-t)
        #         s*J*nearfield(u*dx1+v*dx2,u*dy1+v*dy2,z)
        #     end |>first
        sum(Iterators.product((-1,1),(-1,1))) do (sx,sy)
            wx,wy = x-sx,y-sy; J = abs(wx*wy)
            hcubature(SA[0.,0.],SA[1.,1.];atol) do (u,v)
                x′,y′ = x-wx*u,y-wy*v
                J*source(x′,y′,c)*nearfield(x-x′,y-y′,z)
            end |> first
        end
    else
        hcubature(SA[-1.,-1.],SA[1.,1.];atol) do (x′,y′)
            source(x′,y′,c)*nearfield(x-x′,y-y′,z)
        end |> first
    end
end
∫Pnear(SA[0,-1-1e-3,-0.])
∫Pnear(SA[0,-1+1e-3,-0.])
derivative(z->∫Pnear(SA[0,0,z]),0.)

wₖ(x,y,c;kwargs...) = derivative(z->∫Pwave(SA[x-1,y-1],SA[x+1,y+1],c;z,kwargs...),-0.)
wₙ(x,y,c;atol=1e-4,kwargs...) = derivative(z->∫Pnear(SA[x,y,z],c;atol),-0.)
w(x,y,c;kwargs...) = wₖ(x,y,c;kwargs...)+wₙ(x,y,c;kwargs...)

using ForwardDiff:jacobian
LegendreInfluence(N,ys;w=w,kwargs...) = jacobian(vec->map(y->w(0,y,cmat(vec);kwargs...),ys),ones(SVector{N}))
using LinearAlgebra
N=9; sN=9N
ys = gausslegendre(2sN-1)[1][1:sN]
using JLD2
# Aₙ = LegendreInfluence(N,ys,w=wₙ,atol=1e-6) # expensive
# save_object("Aₙ.jld2",Aₙ) # so store it
Aₙ = load_object("Aₙ.jld2")
Uₙ,Sₙ,Vₙ = svd(Aₙ)
Aₖ = LegendreInfluence(N,ys;w=wₖ,atol=1e-12,α=√(6log(10))/17π)
Uₖ,Sₖ,Vₖ = svd(Aₖ)
A = Aₖ+Aₙ
U,S,V = svd(A)

using Plots
cmap = cgrad(:roma, N, categorical=true);
plot();for i in 1:N
    qᵢ(y) = sum(V[j,i]*ϕ(y,j) for j in 1:N)/√S[i]
    plot!(range(-1,1,300),qᵢ,label="mode $i",c=cmap[i])
end;plot!(xlabel="panel width y",ylabel="normalized source mode qᵢ",ylims=(-0.5,0.5))
plot();for i in 1:N
    wᵢ = U[:,i]*√S[i]
    plot!(ys,wᵢ,label="mode $i",c=cmap[i])
end;plot!(xlabel="panel width y",ylabel="normalized velocity mode wᵢ")

ws = @. ys^2-1
cs = U'ws
plot(ys,ws,label="target");plot!(ys,U*cs,label="projection");plot!(ys,ws .- U*cs,label="error")
vs = V*(cs./S) |> SVector{N}
plot(ys,y->source(0,y,cmat(vs)))