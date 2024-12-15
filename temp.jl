using NeumannKelvin,StaticArrays,TypedTables
function prism(h;q=0.2,Z=1,r=1.2)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    B(θ,r) = SA[0.5r*cos(θ),0.5q*r*sin(θ),-Z]
    dθ = π/round(π*0.5/h) # cosine sampling
    @show π/dθ
    mapreduce(vcat,0.5dθ:dθ:2π) do θ
        dx = dθ*hypot(q*cos(θ),sin(θ))
        i = round(log(1+2Z/dx*(r-1))/log(r)) # geometric growth
        wall = mapreduce(vcat,1:i) do j
            z,dz = -dx*(1-r^j)/(1-r),dx*r^j
            param_props.(S,θ,z+0.5dz,dθ,dz)
        end
        # vcat(wall,[param_props(B,θ,2/3,dθ,1)])
        wall
    end |> Table
end
function prism_force(h;Z=1,U=SVector(-1,0,0),kwargs...)
    panels = prism(h;Z)
	q = influence(panels;kwargs...)\(-Uₙ.(panels;U))
	steady_force(q,panels;U,kwargs...)
end
prism_force(0.05,Z=10;G=kelvin,Fn=0.55)

waterline(p) = abs(p.T₁[3]+p.T₂[3]) ≥ -2p.x[3]
function WL_q(panels;Fn=0.3989)
    x = panels.x |> stack
    WL = findall(waterline,panels)
    q = influence(panels;G=kelvin)\(-Uₙ.(panels;U=SA[-1,0,0]))
    (x[1,WL],q[WL])
end
using Plots
plot()
for h in (0.0125,0.025,0.05,0.1,0.2,0.4)
    plot!(WL_q(prism(h))...,label=h)
end
plot!(ylabel="x")