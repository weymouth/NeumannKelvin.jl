using NeumannKelvin,StaticArrays
using NeumannKelvin: wavelike
using SpecialFunctions
using Plots

x = -20:0.01:-1
plot(x,x->wavelike(x,0.,-0.),xlabel="x",ylabel="W(x,0,0)",label="")
plot!(x,x->4π*bessely1(-x),label="",lc=:grey,ls=:dash)
savefig("wavelike_centerline.png")

x = -40:0.01:-1
plot(x,x->wavelike(x,abs(x/(2√2)),-0.),label="")
plot!(x,x->11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",xlabel="x")
plot!(x,x->-11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",ylabel="W(x,x/2√2,0)")
savefig("wavelike_kelvin_angle.png")

z = @. -abs(log(0.01:0.01:1))
plot(z,z->4exp(z)*√(-π*z),lc=:grey,xlabel="z",ylabel="zW(x,0,z)",label="")
plot!(z,z .* wavelike.(-1.2,0.,z),lc=:darkgreen,label="x=-1.2")
plot!(z,z .* wavelike.(-1,0.,z),lc=:green,label="x=-1")
plot!(z,z .* wavelike.(-0.8,0.,z),lc=:seagreen,label="x=-0.8")
savefig("wavelike_vertical.png")

y = logrange(0.1,5,500)
plot(y,wavelike.(-10.,y,-0.1),lc=:darkgreen,label="z=-0.1")
plot!(y,wavelike.(-10.,y,-0.03),lc=:green,label="z=-0.03")
plot!(y,wavelike.(-10.,y,-0.01),lc=:seagreen,label="z=-0.01")
plot!(xlabel="y",ylabel="W(-1,y,z)")
savefig("wavelike_transverse.png")

using NeumannKelvin: g,dg,nsp,refine_ρ,quadgl,stationary_points,combine
using FastGaussQuadrature, QuadGK, Plots, Roots

function bruteW(x,y)
	Wi(t) = exp(-(1+t^2))*sin(g(x,y,t))
	4quadgk(Wi,-Inf,Inf,rtol=1e-10,atol=1e-10)[1]
end

function precise_rng(x,y,Δg=4π;nleg=0,nkon=Inf,nlag=4,R=8log(10)-1)
    S = stationary_points(x,y); a = S[1]
    ga2b(a,b) = abs(g(x,y,a)-g(x,y,b))
    fz(a,b,f=1) = find_zero(t->ga2b(a,t)-min(Δg,ga2b(a,b)/f),(a,b))
    rngs = if length(S)==1 || S[2]>R
        (fz(a,-R),a),(a,fz(a,R))
    else
        b = S[2]
        (fz(a,-R),fz(a,b,2)),(fz(b,a,2),fz(b,R))
    end 

    # Compute real-line contributions
    ĝ(t)=g(x,y,t)+im*(1+t^2)
    dĝ(t)=dg(x,y,t)+2im*t
    function f(t)
        u,v = reim(ĝ(t))
        exp(-v)*sin(u)
    end
    I = if nleg>0 # use fixed-size Gauss-Legendre
        xleg,wleg = gausslegendre(nleg)
        sum(quadgl(f,rng...;x=xleg,w=wleg) for rng in rngs)
    else          # use adaptive Gauss-Konrad
        sum(quadgk(f,rng...;atol=1e-10,maxevals=15*nkon)[1] for rng in rngs)
    end

    # Add the end point contributions
    function nsp(h₀)
        s,g₀,h = 0.,ĝ(h₀),h₀+0im
        for (p,w) in zip(gausslaguerre(nlag)...)
            h = find_zero((h->ĝ(h)-g₀-im*p,h->dĝ(h)),h,Roots.Newton())
            s += w*imag(exp(im*g₀)*im/dĝ(h))
        end;s
    end
    for rng in combine(rngs...), (i,t₀) in enumerate(rng)
        I += (-1)^i*nsp(t₀)
    end
    4I
end

Δg = 0.3:0.05:6pi
cmap = cgrad(:matter, 5, categorical = true);
x,y = -110.,30.
plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for (i,np) in enumerate((1,3,7))
    plot!(Δg,g->abs(precise_rng(x,y,g;nkon=np)-bruteW(x,y)),label="2*15*$np",c=cmap[i]);
end;plot!(legendtitle="Gauss-Konrad points",legendtitlefontsize=9)
savefig("Gauss-Konrad points.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for (i,np) in enumerate((15,20,25))
    plot!(Δg,g->abs(precise_rng(x,y,g;nleg=np)-bruteW(x,y)),label="2*$np",c=cmap[i]);
end;plot!(legendtitle="Gauss-Legendre points",legendtitlefontsize=9)
savefig("Gauss-Legendre points.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for (i,np) in enumerate(2:6)
    plot!(Δg,g->abs(precise_rng(x,y,g;nlag=np)-bruteW(x,y)),label=np,c=cmap[i]);
end;plot!(legendtitle="Gauss-Laguerre points",legendtitlefontsize=9)
savefig("Gauss-Laguerre points.png")

plot(xlabel="Δg",ylabel="|error|",yscale=:log10,ylim=(1e-10,1));
for i = 1:5
    R = 2^(2i-2)
    x,y = -11R/hypot(11,3),3R/hypot(11,3)
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