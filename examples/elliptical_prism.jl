using NeumannKelvin,Plots

# Neumann-Kelvin wave drag on the half-body, doubled for both sides
Cw(panels;domain) = -2steadyforce(directsolve!(PanelSystem(panels;domain),verbose=false),S=1)[1]

# Thin-ship wave drag: prescribed sources q=nₓ/4π on `sources` (no solve), pressure measured on the hull
function Cw₀(panels,sources=panels;domain)
    sys = NeumannKelvin.set_q!(PanelSystem(sources;domain),components(panels.n,1)/4π)
    2sum(p->cₚ(p.x,sys)*p.n[1]*p.dA,panels)
end
squash(v::SVector{3,<:Number}) = v .* SA[1,0,1] # project onto y=0
squash(v::AbstractArray) = map(squash,v)
squash(v) = v
centreplane(panels) = Table(map(p->map(squash,p),panels)) # Michell's thin-ship sources

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
    measure.(surface,uc,vc',du,dv',flip=true) |> Table
end
panels = prism(0.05)

dat = map(logrange(0.2,0.6,40)) do Fn
    domain = Kelvin(Fn^2,sym_axes=2,filter=false)
    (Fn=Fn,Contour=Cw(panels,domain=Kelvin(Fn^2,sym_axes=2,contour=true,filter=false)),
           NoContour=Cw(panels;domain),
           ThinShip=Cw₀(panels,centreplane(panels);domain),
           ThinSource=Cw₀(panels;domain))
end |> Table
plot(dat.Fn,[dat.Contour dat.NoContour dat.ThinShip dat.ThinSource],label=["Contour" "NoContour" "ThinShip" "ThinSource"],
     xlabel="Fn",ylabel="R/½ρU²L²",size=(400,500),ylims=(0,0.1))
savefig("examples/elliptical_prism.png")

function WL(panels;domain) # wave elevation ζg/U² along the waterline
    sys = directsolve!(PanelSystem(panels;domain),verbose=false)
    wl = filter(NeumannKelvin.onwaterline,panels)
    components(wl.x,1),[ζ(x .* SA[1,1,0],sys) for x in wl.x]
end
plot(); for (n,c) in zip((40,28,20),reverse(palette(:Blues_4)))
    x,z = WL(prism(1/n,1/2n); domain=Kelvin(0.2^2,sym_axes=2,contour=true,filter=true))
    plot!(x,z,label="hᵤ=L/$n",line=(2),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζg/U²")
savefig("examples/elliptical_prismWLconv.png")
