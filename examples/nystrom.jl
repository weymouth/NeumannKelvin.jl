using FastGaussQuadrature,StaticArrays,LinearAlgebra,NeumannKelvin,ThreadsX,ForwardDiff,QuadGK,IntervalSets,SpecialFunctions
using SpecialFunctions: sphericalbesselj
jₘ(m,x) = m==0 ? sinc(x/π) : sign(x)^m*sphericalbesselj(m,abs(x))
basis(x,M) = SVector{M}([im^m*jₘ(m,x) for m in 0:M-1])
using NeumannKelvin:kₓ,g
function ∫wavelike(z,a,b,c,d;atol=1e-4,T=13,M=1,N=2)
    a≥0 && return zero(SMatrix{M,N,typeof(z)})  # Panel downstream
    b = min(-eps(),b)                           # Heaviside limit
    Δx, Δy, mx, my = b-a, d-c, (a+b)/2, (c+d)/2
    @fastmath f(t) = imag(basis(Δx*kₓ(t)/2,M)*basis(Δy*t*kₓ(t)/2,N)'*exp(im*g(mx,my,t)+z*(1+t^2)))
    4Δx*Δy*quadgk(f,-T,T;atol)[1]
end

# nearfield & wavelike (kelvin) ∂/∂z operators
z_dual = ForwardDiff.Dual(-0.,1.)
N(x,y) = NeumannKelvin.nearfield(x,y,z_dual)
∫W(a,b,c,d;kwargs...) = ∫wavelike(z_dual,a,b,c,d;kwargs...)
N(1.,1.),∫W(-1.,1.,-1.,0.) # initialize

# Nyström evaluation points & patch boundaries
nq = 64
bnds,_ = gausslobatto(nq+1)
xs = [0.5*(bnds[i]+bnds[i+1]) for i in 1:nq]
Xij = [(i,j) for i in 1:nq for j in 1:nq]
Qij = [(i,j) for i in 1:nq for j in 1:(nq-1)]
Qmap(i,j) = (i-1)*(nq-1) + j
dx = diff(bnds); da = [dx[i]*dx[j] for (i,j) in Xij]; wght,iwght = diagm(sqrt.(da)), diagm(inv.(sqrt.(da)))

# Halve matrix due to y-symmetry
modes = [(i,j) for i in 1:nq for j in 1:nq÷2]
Φᵢ = [(i==m && (j==n || j==nq-n)) ? 1 : 0 for (i,j) in Qij, (m,n) in modes]
Φₒ = [(i==m && (j==n || j==nq-n+1)) ? 1 : 0 for (i,j) in Xij, (m,n) in modes]

# Create operator and fix the diagonal
using JLD2
A = try
    load_object("influenceA$nq.jld2")
catch
    A = Matrix{ForwardDiff.Dual{Nothing,Float64,1}}(undef,length(Xij),length(Qij))
    ThreadsX.foreach(enumerate(Xij)) do (I,(i,j))
        for (I′,(i′,j′)) in enumerate(Xij)
        x,y,x′,y′ = xs[i],xs[j],xs[i′],xs[j′]
        a,b,c,d = bnds[i′],bnds[i′+1],bnds[j′],bnds[j′+1]
        L₀ₖ,L₁ₖ = ∫W(x-b,x-a,y-d,y-c)
        xrange,yrange = I′==I ? ((a,x,b),(c,y,d)) : ((a,b),(c,d))
        L₀ₙ,L₁ₙ = (abs(I′-I)>2 ? SA[1,0]*N(x-x′,y-y′)*da[I′] :
            quadgl(x′->quadgl(y′->SA[1,(2y′-d-c)/(d-c)]*N(x-x′,y-y′),yrange...),xrange...))
        j′ < nq && (A[I,Qmap(i′,j′)]   = (L₀ₖ+L₀ₙ+L₁ₖ+L₁ₙ)/2)  # edge above j′
        j′ > 1  && (A[I,Qmap(i′,j′-1)] = (L₀ₖ+L₀ₙ-L₁ₖ-L₁ₙ)/2)  # edge below j′
    end;end
    Φₒ'*A*Φᵢ
end
save_object("influenceA$nq.jld2",A)

# Build full basis for ϕ
U,S,V = svd(ForwardDiff.value.(A))
@assert ForwardDiff.value.(A)*V ≈ U*diagm(S)

# 99.5% energy cutoff
Eᵣ = cumsum(S.^2)./sum(S.^2)
k = findfirst(Eᵣ .> 0.995)
@show k,Eᵣ[k],S[1]/S[k]
using Plots
plot(S,ylabel="σ",label=nothing);scatter!([k],S[k:k],label="cutoff",xscale=:log10,yscale=:log10,ylims=(1e-5*S[1],S[1]))

# Take a look at the modes
function plotmodes(V,k;m=nq,n=nq,y=xs,x=xs)
    ks = (1:6...,k-2:k...)
    vs = [reshape(V[:,k], (m,n)) for k in ks]
    maxval = maximum(abs, vcat(vs...))  # shared scale
    plots = [contourf(x, y, v;
            aspect_ratio = :equal,
            color = :balance,
            title = "Mode $k", titlefontsize=10,
            clims = (-maxval,maxval),levels = 16,
            axis=false, legend=false, framestyle=:none)
        for (k,v) in zip(ks,vs)]
    plot(plots..., layout = (3, 3), size=(800,800))
end
plotmodes(Φᵢ*V,max(9,k),m=nq-1,y=bnds[2:end-1])
savefig("qmodes$nq.png")
plotmodes(Φₒ*U,max(9,k))
savefig("pmodes$nq.png")

# Build smooth truncated basis for w
k = max(9,k)
Aₖ = ForwardDiff.partials.(A) .|> first
Y = Aₖ*V[:,1:k]
plotmodes(Φₒ*Y,k)
savefig("ymodes$nq.png")
Uₖ,Sₖ,Vₖ = svd(Aₖ*V[:,1:k]); Vₖ = V[:,1:k]*Vₖ
@assert Aₖ*Vₖ ≈ Uₖ*diagm(Sₖ)
plotmodes(Φᵢ*Vₖ,k,m=nq-1,y=bnds[2:end-1])
savefig("wqmodes$nq.png")
plotmodes(Φₒ*Uₖ,k)
savefig("wmodes$nq.png")
