using NeumannKelvin,Plots,ColorSchemes
# Hull
wigley(hᵤ;B=0.125,D=0.05,hᵥ=hᵤ/D) = measure_panel.(
    (u,v)->SA[u-0.5,2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ,flip=true) |> Table
∫kelvin₂ = reflect(∫kelvin,2)
∫kelvin₂(SA[-1,0,-0],wigley(0.05)[1];contour=true); # Initiate Chebychev polynomials

# Convergence test at low Fn for large B/L (hardest case)
Fn,B,D=0.2,1/8,1/16; ps = (ϕ=∫kelvin₂,ℓ=Fn^2,contour=true)
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (n,c) in zip((32,48,64,80),colorschemes[:Blues_4][1:end])
    h = 1/n; panels = wigley(h;B,D)
    q = influence(panels;ps...)\components(panels.n,1)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    z = map(xy->2ζ(xy[1],xy[2],q,panels;ps...),zip(x,y))
    plot!(x,z,label="h/L=1/$n",line=(2),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζg/U²",title="Wigley B/L=$B, Fn=$Fn",ylims=(-0.2,0.4))
savefig("examples/WigleyWLconv.png")

panels = wigley(1/80;B,D,hᵥ=1/8);
x,y,_ = filter(onwaterline,panels) |> p -> components(p.x);
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (Fn,c) in zip((0.2,0.316,0.4),colorschemes[:Blues_3])
    ps = (ϕ=∫kelvin₂,ℓ=Fn^2,contour=true)
    q = influence(panels;filter=false,ps...)\components(panels.n,1)
    z = map(xy->2ζ(xy[1],xy[2],q,panels;filter=false,ps...),zip(x,y))
    plot!(x,z,label="Fn=$Fn",line=(2,:dash);c)
    q = influence(panels;ps...)\components(panels.n,1)
    z = map(xy->2ζ(xy[1],xy[2],q,panels;ps...),zip(x,y))
    plot!(x,z,label="z_max filter",line=(2);c)
end;plot!(xlabel="x/L",ylabel="ζg/U²",title="Wigley B/L=$B",ylims=(-0.2,0.4))
savefig("examples/WigleyWLfilter.png")