using NeumannKelvin,Plots
∫kelvin₂ = reflect(∫kelvin,2)
Cw(panels;kwargs...) = steady_force(influence(panels;kwargs...)\first.(panels.n),panels;kwargs...)[1]

∫kelvin₀(x,p;kwargs...) = 2∫kelvin(x,reflect(p,SA[1,0,1]);kwargs...)
Cw₀(panels;kwargs...) = steady_force(first.(panels.n)/2π,panels;kwargs...)[1]

R = 10; S(θ,z) = SA[R*cos(θ),0.2R*sin(θ),z]
dat2 = map(logrange(0.2,0.6,35)) do Fn
    (Fn=Fn,Cw=Cw(cylinder,ϕ=∫kelvin₂,ℓ=Fn^2*2R,contour=true,filter=false)/4R^2)
end |> Table
dat3 = map(logrange(0.2,0.6,35)) do Fn
    (Fn=Fn,Cw=Cw₀(cylinder,ϕ=∫kelvin₀,ℓ=Fn^2*2R)/4R^2)
end |> Table
dat4 = map(logrange(0.2,0.6,35)) do Fn
    (Fn=Fn,Cw=Cw₀(cylinder,ϕ=∫kelvin₂,ℓ=Fn^2*2R)/4R^2)
end |> Table

plot(dat.Fn,dat2.Cw,label="Present");
plot!(dat.Fn,dat3.Cw,label="Thin-ship");
plot!(dat.Fn,dat4.Cw,label="Thin-source")
savefig("examples/ellipical_prism.png")
