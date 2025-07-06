using NeumannKelvin,Plots,ColorSchemes
using NeumannKelvin: kelvin
function ∫contour(ξ,p;Fn,z_max=-1/50Fn^2)
    dx = extent.(components(p.xᵤᵥ)) .* SA[1,1,0]
    n₁dx₂ = -dx[2]^2/norm(dx)
    x = p.x .* SA[1,1,0]
    G = 0.5sum(α->kelvin(ξ,x+α*dx;Fn,z_max),(-0.5/√3,0.5/√3))
    Fn^2*G*n₁dx₂
end
function ∫surface(x,p;Fn,χ=false)
    (!χ || !onwaterline(p)) && return ∫kelvin(x,p;Fn) # no waterline
    ∫kelvin(x,p;Fn)+∫contour(x,p;Fn)
end
∫surface₂(x,p;kwargs...)=∫surface(x,p;kwargs...)+∫surface(x,reflect(p,2);kwargs...)

# Hull and lid
wigley(hᵤ;B=0.125,D=0.05,hᵥ=hᵤ/D) = measure_panel.(
    (u,v)->SA[u-0.5,2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ,flip=true) |> Table
∫kelvin(SA[-1,0,-0],wigley(0.05)[1]); # Initiate Chebychev polynomials

# Convergence test at low Fn (hardest case)
Fn,B,D=0.2,1/10,1/16
plot(Shape([-0.5,0.5,0.5,-0.5],[-1,-1,1,1]),opacity=0.5,c=:grey,linewidth=0,label="");
for (n,c) in zip((32,48,64,80),colorschemes[:Blues_4][1:end])
    h = 1/n; panels = wigley(h;B,D)
    q = influence(panels;ϕ=∫surface₂,Fn)\components(panels.n,1)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    z = map(xy->2ζ(xy[1],xy[2],q,panels;ϕ=∫surface₂,Fn),zip(x,y))
    plot!(x,z,label="h/L=1/$n",line=(2,:dash),marker=(2,),markerstrokewidth=0;c)
end;plot!(xlabel="x/L",ylabel="ζg/U²",title="Wigley B/L=$B, Fn=$Fn",ylims=(-0.2,0.4))
savefig("examples/WigleyWL.png")