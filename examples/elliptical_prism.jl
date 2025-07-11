using NeumannKelvin,Plots,ColorSchemes
∫kelvin₂ = reflect(∫kelvin,2)
Cw(panels;kwargs...) = 2steady_force(influence(panels;kwargs...)\first.(panels.n),panels;kwargs...)[1]

∫kelvin₀(x,p;kwargs...) = 2∫kelvin(x,reflect(p,SA[1,0,1]);kwargs...)
Cw₀(panels;kwargs...) = 2steady_force(first.(panels.n)/4π,panels;kwargs...)[1]

function prism(hᵤ,hᵥ=1.618hᵤ;q=0.2,Z=2,c=0.05,r=1.2)
    # Define prism surface
    surface(u,v) = SA[0.5cos(π*u),0.5q*sin(π*u),-v]
    # Define WL segments based on pseudo-arclength
    S,s⁻¹ = NeumannKelvin.arclength(u->surface(u,0),hᵤ,c,0,1)
    Nᵤ = round(Int,S/hᵤ)
    u = s⁻¹(range(0,S,Nᵤ+1))
    uc,du = 0.5*(u[1:end-1]+u[2:end]), diff(u)
    # Define vertical segments with geometric growth
    Nᵥ = round(Int,log(1+Z/hᵥ*(r-1))/log(r))
    v = @. hᵥ*(1-r^(0:Nᵥ))/(1-r)
    vc,dv = 0.5*(v[1:end-1]+v[2:end]), diff(v)
    # Measure each panel
    measure_panel.(surface,uc,vc',du,dv',flip=true) |> Table
end
panels = prism(0.05)

dat = map(logrange(0.2,0.6,40)) do Fn
    (Fn=Fn,Contour=Cw(panels,ϕ=∫kelvin₂,ℓ=Fn^2,contour=true,filter=false),
           NoContour=Cw(panels,ϕ=∫kelvin₂,ℓ=Fn^2),
           ThinShip=Cw₀(panels,ϕ=∫kelvin₀,ℓ=Fn^2),
           ThinSource=Cw₀(panels,ϕ=∫kelvin₂,ℓ=Fn^2))
end |> Table
plot(dat,ylabel="R/½ρU²L²",size=(400,500),ylims=(0,0.1))
savefig("examples/ellipical_prism.png")

function WL(panels;kwargs...)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    q = influence(panels;kwargs...)\components(panels.n,1)
    z = map(xy->ζ(xy[1],xy[2],q,panels;kwargs...),zip(x,y))
    x,z,q
end
plot(); for (n,c) in zip((40,28,20),reverse(colorschemes[:Blues_4]))
    panels = prism(1/n,1/2n)
    x,z = WL(panels; ϕ=∫kelvin₂, ℓ=0.2^2, contour=true, filter=true)
    plot!(x,z,label="hᵤ=L/$n",line=(2),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζg/U²")
savefig("examples/ellipical_prismWLconv.png")