using NeumannKelvin,StaticArrays
using Plots

x = -20:0.01:-1
plot(x,x->NeumannKelvin.wavelike(x,0.,-0.),xlabel="x",ylabel="W(x,0,0)",label="")
plot!(x,x->4π*bessely1(-x),label="",lc=:grey,ls=:dash)
savefig("wavelike_centerline.png")

x = -40:0.01:-1
plot(x,x->NeumannKelvin.wavelike(x,abs(x/(2√2)),-0.),label="")
plot!(x,x->11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",xlabel="x")
plot!(x,x->-11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",ylabel="W(x,x/2√2,0)")
savefig("wavelike_kelvin_angle.png")

z = log.(0.01:0.01:1)
plot(z,z->4exp(z)*√(-π*z),lc=:grey,xlabel="z",ylabel="zW(x,0,z)",label="")
plot!(z,z .*NeumannKelvin.wavelike.(-1.2,0.,z),lc=:darkgreen,label="x=-1.2")
plot!(z,z .*NeumannKelvin.wavelike.(-1,0.,z),lc=:green,label="x=-1")
plot!(z,z .*NeumannKelvin.wavelike.(-0.8,0.,z),lc=:seagreen,label="x=-0.8")
savefig("wavelike_vertical.png")

using NeumannKelvin: g,dg,nsp,refine_ρ,quadgl_ab,stationary_points
using FastGaussQuadrature, QuadGK, Plots

function bruteW(x,y)
	Wi(t) = exp(-(1+t^2))*sin(g(x,y,t))
	4quadgk(Wi,-Inf,Inf,rtol=1e-10,atol=1e-10)[1]
end

function precise_rng(x,y,Δg=4π;nleg=20,nkon=Inf,nlag=5,R=8log(10)-1)
    S = stationary_points(x,y)
    a = S[1]
    ρ₀ = Δg*√(inv(0.5Δg*abs(x)+y^2)+inv(x^2+Δg*abs(y)))
    ρₐ = refine_ρ(a,t->g(x,y,t),t->dg(x,y,t),ρ₀;s=-1,Δg,itmx=30,rtol=1e-4)
    # rngs,mid = [(a-ρₐ,a),(a,a+ρₐ)],a
    rngs = ((a-ρₐ,a+ρₐ),)
 
    if length(S)==2 && S[2]<R
        b = S[2]
        ρᵦ = refine_ρ(b,t->g(x,y,t),t->dg(x,y,t),ρ₀;s= 1,Δg,itmx=30,rtol=1e-4)
        # rngs = [(a-ρₐ,a+ρₐ),(b-ρᵦ,b+ρᵦ)]
        # mid = (rngs[1][2]+rngs[2][1])/2
        # (rngs[1][2]≥rngs[2][1] || 2abs(g(x,y,a)-g(x,y,b))≥Δg) && (rngs = [(rngs[1][1],mid),(mid,rngs[2][2])])
        rngs = (rngs...,(b-ρᵦ,b+ρᵦ))
        (rngs[1][2]≥rngs[2][1] || 2abs(g(x,y,a)-g(x,y,b))≥Δg) && (rngs = ((rngs[1][1],rngs[2][2]),))
    end

    # Compute real-line contributions
    ĝ(t)=g(x,y,t)+im*(1+t^2)
    dĝ(t)=dg(x,y,t)+2im*t
    @fastmath @inline function f(t)
        u,v = reim(ĝ(t))
        exp(-v)*sin(u)
    end
    # xleg,wleg = gausslegendre(nleg)
    # I = sum(quadgl_ab(f,rng...;x=xleg,w=wleg) for rng in rngs)
    I = sum(quadgk(f,rng...;order=8,atol=1e-10,maxevals=2*nkon+1)[1] for rng in rngs)

    # Add the end point contributions
    xlag,wlag = gausslaguerre(nlag)
    for rng in rngs, (i,t₀) in enumerate(rng)
        # t₀ ≠ mid && (I+= (-1)^i*nsp(t₀,ĝ,dĝ;xlag,wlag))
        I+= (-1)^i*nsp(t₀,ĝ,dĝ;xlag,wlag)
    end
    4I
end

Δg = 0.3:0.05:6pi
cmap = cgrad(:matter, 5, categorical = true);
x,y = -110,30
plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for (i,np) in enumerate((8,16,32,64))
    plot!(Δg,g->abs(precise_rng(x,y,g;nkon=np)-bruteW(x,y)),label=np,c=cmap[i]);
end;plot!(legendtitle="Gauss-Konrad points",legendtitlefontsize=9)
savefig("Gauss-Konrad points.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for (i,np) in enumerate(2:6)
    plot!(Δg,g->abs(precise_rng(x,y,g;nlag=np)-bruteW(x,y)),label=np,c=cmap[i]);
end;plot!(legendtitle="Gauss-Laguerre points",legendtitlefontsize=9)
savefig("Gauss-Laguerre points.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for i = 1:5
    R = 2^(2i-2)
    x,y = -11R/11.4,3R/11.4
    plot!(Δg,g->abs(precise_rng(x,y,g)-bruteW(x,y)),label=R,c=cmap[i])
end;plot!(legendtitle="scaled distance",legendtitlefontsize=9)
savefig("wave mid.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for i = 1:5
    R = 2^(2i-2)
    x,y = -R,R/√8
    plot!(Δg,g->abs(precise_rng(x,y,g)-bruteW(x,y)),label=R,c=cmap[i])
end;plot!(legendtitle="scaled distance")
savefig("wake edge.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for i = 1:5
    R = 2^(2i-2)
    x,y = -R,R/2
    plot!(Δg,g->abs(precise_rng(x,y,g)-bruteW(x,y)),label=R,c=cmap[i])
end;plot!(legendtitle="scaled distance")
savefig("wake outer.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for i = 1:5
    R = 2^(2i-2)
    x,y = -R,0.
    plot!(Δg,g->abs(precise_rng(x,y,g)-bruteW(x,y)),label=R,c=cmap[i])
end;plot!(legendtitle="scaled distance")
savefig("wake center.png")