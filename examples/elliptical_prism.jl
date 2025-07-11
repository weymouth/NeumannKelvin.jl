using NeumannKelvin,Plots
∫kelvin₂ = reflect(∫kelvin,2)
Cw(panels;kwargs...) = 2steady_force(influence(panels;kwargs...)\first.(panels.n),panels;kwargs...)[1]

∫kelvin₀(x,p;kwargs...) = 2∫kelvin(x,reflect(p,SA[1,0,1]);kwargs...)
Cw₀(panels;kwargs...) = 2steady_force(first.(panels.n)/4π,panels;kwargs...)[1]

function prism(h;q=0.2,Z=1,r=1.2)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    dθ = π/round(π*0.5/h) # cosine sampling
    mapreduce(vcat,0.5dθ:dθ:π) do θ
        dx = 1.618dθ*hypot(q*cos(θ),sin(θ))
        i = round(log(1+2Z/dx*(r-1))/log(r)) # geometric growth
        mapreduce(vcat,1:i) do j
            z,dz = -dx*(1-r^j)/(1-r),dx*r^(j-1)
            measure_panel.(S,θ,z+0.5dz,dθ,dz)
        end
    end |> Table
end
panels = prism(0.05)

dat = map(logrange(0.2,0.6,40)) do Fn
    (Fn=Fn,Cw=Cw(panels,ϕ=∫kelvin₂,ℓ=Fn^2))
end |> Table
dat2 = map(logrange(0.2,0.6,40)) do Fn
    (Fn=Fn,Cw=Cw(panels,ϕ=∫kelvin₂,ℓ=Fn^2,contour=true,filter=false))
end |> Table
dat3 = map(logrange(0.2,0.6,40)) do Fn
    (Fn=Fn,Cw=Cw₀(panels,ϕ=∫kelvin₀,ℓ=Fn^2))
end |> Table
dat4 = map(logrange(0.2,0.6,40)) do Fn
    (Fn=Fn,Cw=Cw₀(panels,ϕ=∫kelvin₂,ℓ=Fn^2))
end |> Table

plot(dat.Fn,dat.Cw,label="No-contour");
plot!(dat.Fn,dat2.Cw,label="Contour");
plot!(dat.Fn,dat3.Cw,label="Thin-ship");
plot!(dat.Fn,dat4.Cw,label="Thin-source")
savefig("examples/ellipical_prism.png")
