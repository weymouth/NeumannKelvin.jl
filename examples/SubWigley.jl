using NeumannKelvin,Plots,ColorSchemes
using NeumannKelvin: kelvin
# Hull and lid
wigley(hᵤ;B=0.125,D=0.05,hᵥ=hᵤ/D) = measure_panel.(
    (u,v)->SA[u-0.5,2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ,flip=true) |> Table
∫kelvin(SA[-1,0,-0],wigley(0.05)[1]); # Initiate Chebychev polynomials

∫kelvin₂(x,p;kwargs...) = ∫kelvin(x,p;kwargs...)+∫kelvin(x,reflect(p,2);kwargs...)
∫G₂(x,p;kwargs...) = ∫G(x,p;kwargs...)+∫G(x,reflect(p,2);kwargs...)
∫G₂₃(x,p;kwargs...) = ∫G₂(x,p;kwargs...)+∫G₂(x,reflect(p,3);kwargs...)
function ∫contour(ξ,p;Fn)
    dx = extent.(components(p.xᵤᵥ)) .* SA[1,1,0]
    dl = norm(dx)
    x = p.x .* SA[1,1,0]
    -Fn^2*dx[2]^2/2dl*sum(α->kelvin(ξ,x+α*dx;Fn),(-0.5/√3,0.5/√3))
    # -Fn^2*dx[2]^2/2dl*kelvin(ξ,p.x .* SA[1,1,0];Fn)
end
function ∫surface(x,p;Fn,χ=true,d²=4)
    (!χ || !onwaterline(p)) && return ∫kelvin(x,p;d²,Fn) # no waterline
    ∫kelvin(x,p;d²,Fn)+∫contour(x,p;Fn)
end
∫surface₂(x,p;kwargs...)=∫surface(x,p;kwargs...)+∫surface(x,reflect(p,2);kwargs...)

# Convergence test at low Fn (hardest case)
using NeumannKelvin:Φ
Fn,B,D=0.2,1/10,1/16
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (n,c) in zip((32,48,64,80),colorschemes[:Blues_4][1:end])
    h = 1/n; panels = wigleyhull(h;B,D)
    q = influence(panels;ϕ=∫surface₂,Fn)\components(panels.n,1)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    # w = map(xy->derivative(z->Φ(SA[xy[1],xy[2],z],q,panels;ϕ=∫kelvin₂),0),zip(x,y))
    # plot!(x,w,label="h/L=1/$n",line=(2,:dash),marker=(2,),markerstrokewidth=0;c)
    z = map(xy->2ζ(xy[1],xy[2],q,panels;ϕ=∫surface₂,Fn),zip(x,y))
    plot!(x,z,label="h/L=1/$n",line=(2,:dash),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="w/U",title="Wigley B/L=$B, Fn=$Fn",ylims=(-0.2,0.4))
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
