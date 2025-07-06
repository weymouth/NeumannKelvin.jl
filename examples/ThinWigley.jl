using NeumannKelvin,Plots,ColorSchemes

# Thin-ship theory projects allocation points on the centerplane
wigley(hᵤ;B=0.01,D=0.05,hᵥ=hᵤ/D) = measure_panel.(
    (u,v)->SA[u-0.5,2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ,flip=true) |> Table
∫kelvin₂(x,p;kwargs...) = ∫kelvin(x,p;kwargs...)+∫kelvin(x,reflect(p,2);kwargs...)
∫kelvin₂(SA[-1,0,-0],wigley(0.05)[1]); # Initiate Chebychev polynomials

# Convergence test at low Fn (hardest case)
ps = (ϕ=∫kelvin₂,Fn=0.2); B=0.125
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="",ylims=(-0.1,0.15));
for (n,c) in zip((20,40,60),colorschemes[:Blues_4][2:end])
    h = 1/n; panels = wigley(h;B)
    x = filter(onwaterline,panels) |> p -> components(p.x,1)
    x = [-reverse((0.5+0.5h):h:2);x;(0.5+0.5h):h:1]
    q = components(panels.n,1)/4π;
    plot!(x,x->ζ(x,0,q,panels;ps...)*ps.Fn^2/B,label="h/L=1/$n",
        line=(2,:dash),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζ/B",title="Thin-Ship Wigley B=L/8, Fn=$(ps.Fn)")
savefig("examples/thinWigleyWL.png")

begin
    h = 1/40; panels = wigley(h;B)
    q = components(panels.n,1)/4π;
    contour(-2:h:1,-1:0.5h:1,(x,y)->ζ(x,y,q,panels;ps...)*ps.Fn^2/B,
            cmap=:seismic,clims=(-0.07,0.07),levels=(10))
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    plot!(Shape([x;reverse(x)],[y;-reverse(y)]),c=:grey,linewidth=0,label="")
end;plot!(aspectratio=:equal,xlabel="x/L",ylabel="y/L",colorbartitle="ζ/B")
savefig("examples/thinWigleyζ.png")

Cw(panels;kwargs...) = steady_force(components(panels.n,1)/4π,panels;kwargs...)[1]
data = map(Fn->(Fn=Fn,Cw=Cw(panels;ps...,Fn)/0.5B^2),logrange(0.2,1.2,120)) |> Table
plot(data.Fn,data.Cw,label="",xlabel="Fn",ylabel="R/½ρU²B²",ylims=(0,0.05)) 
savefig("examples/thinWigleyCw.png")
