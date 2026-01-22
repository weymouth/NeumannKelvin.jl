using NeumannKelvin,Plots,ColorSchemes
# Hull
wigley(hᵤ;B=1/8,D=1/16,hᵥ=0.5hᵤ/D) = measure.(
    (u,v)->SA[u-0.5,-2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ) |> Table

NKsys = NKPanelSystem(wigley(0.025);ℓ=1/2π,sym_axes=2,contour=true) |> directsolve!
viz(NKsys)

# Get water line points
function WL(panels;kwargs...)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    q = influence(panels;kwargs...)\components(panels.n,1)
    z = map(xy->ζ(xy[1],xy[2],q,panels;kwargs...),zip(x,y))
    x,z
end

# Convergence test at low Fn for large B/L (hardest case)
Fn = 0.2; ∫kelvin₂ = reflect(∫kelvin,2)
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (n,c) in zip((32,48,64,80),colorschemes[:Blues_4][1:end])
    x,z = WL(wigley(1/n); ϕ=∫kelvin₂, ℓ=Fn^2, contour=true)
    plot!(x,2z,label="h=L/$n",line=(2),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζg/U²",title="Wigley, Fn=$Fn",ylims=(-0.2,0.4))
savefig("examples/WigleyWLconv.png")

panels = wigley(1/80;hᵥ=1/8);
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (Fn,c) in zip((0.2,0.316,0.4),colorschemes[:Blues_3])
    x,z = WL(panels; ϕ=∫kelvin₂, ℓ=Fn^2, contour=true, filter=false)
    plot!(x,2z,label="Fn=$Fn",line=(2,:dash);c)
    x,z = WL(panels; ϕ=∫kelvin₂, ℓ=Fn^2, contour=true)
    plot!(x,2z,label="z_max filter",line=(2);c)
end;plot!(xlabel="x/L",ylabel="ζg/U²",title="Wigley hᵤ=L/80, hᵥ=D/8",ylims=(-0.2,0.4))
savefig("examples/WigleyWLfilter.png")