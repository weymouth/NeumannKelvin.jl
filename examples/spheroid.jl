using NeumannKelvin
function spheroid(h;Z=-0.5,L=1,r=0.25)
	S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
	dθ₁ = π/round(π*0.5L/h)
	odd = true
	mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
		dθ₂ = 2π/max(3,round(2π*r*sin(θ₁)/h))
		odd ? (θ₂₁=0.5dθ₂;odd=false) : (θ₂₁=0;odd=true)
		param_props.(S,θ₁,θ₂₁:dθ₂:2π-0.1dθ₂,dθ₁,dθ₂)
	end |> Table
end

# Sphere area and added mass convergence
using LinearAlgebra,Plots
r = 0.5
A,V = 4π*r^2,4π/3*r^3
dat = map(0.5 .^(1:6)) do h
	panels = spheroid(h;Z=0,r)
	Aerror = sum(panels.dA)/A-1
	Merror = added_mass(panels)./0.5V-I
	(log2h=log2(h),normAerror=norm(Aerror),normMerror=norm(Merror))
end |> Table
using CSV
CSV.write("sphere_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normAerror),label="||A error||")
plot!(dat.log2h,log2.(dat.normMerror),label="||M error||")
plot!(xlabel="log₂ h/D",ylabel="log₂ error",title="Sphere convergence")
savefig("sphere_convergence.png")

# 6:1 Spheroid added_mass convergence
r = 0.5/6.01
V = 4π/3*0.5*r^2; sol = [0.045 0 0; 0 0.918 0; 0 0 0.918]
dat = map(0.5 .^(0:0.5:3)) do h
	panels = spheroid(h*2r;Z=0,r)
	Merror = added_mass(panels)./V-sol
	(log2h=log2(h),normMerror=norm(Merror))
end |> Table
CSV.write("spheroid_ma_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normMerror),label="")
plot!(xlabel="log₂ h/2b",ylabel="log₂ ||M error||",title="6:1 spheroid M convergence")
savefig("spheroid_ma_convergence.png")

# Spheroid added_mass sweep
dat = map(0.025:0.025:0.75) do r
	panels = spheroid(min(0.5r,1/8);Z=0,r)
	V = 4π/3*0.5*r^2
	M = added_mass(panels)./V
	(r=r,M₁₁=M[1,1],M₂₂=M[2,2],M₃₃=M[3,3])
end |> Table
CSV.write("spheroid_ma_sweep.csv",dat)
plot(2dat.r,dat.M₁₁,label="M₁₁/V")
plot!(2dat.r,dat.M₂₂,label="M₂₂/V",ylims=(0,1))
plot!(2dat.r,dat.M₃₃,label="M₃₃/V")
plot!(xlabel="b/a",title="Spheroid M components")
savefig("spheroid_ma_sweep.png")

function surface_spheroid(h,r)
	S(θ₁,θ₂) = SA[0.5cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)]
	dθ₁ = π/round(π*0.5/h) # azimuth step size
	dθ₂ = π/round(π*r/h)   # constant polar step size (classic)
    x,y,z = eachrow(S.(0.5dθ₁:dθ₁:π,0)|>stack) # WL
	x,y,mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
		dθ₂ = π/max(2,round(2π*r*sin(θ₁)/h)) # adaptive polar step
		param_props.(S,θ₁,π+0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
	end |> Table
end

plot(); r = 0.5/6.01;
for h in 2r .* (0.5 .^ (1.75:0.25:2.75))
# for h in 2r .* (0.5 .^ (2:0.25:3))
	xp,yp,panels = surface_spheroid(h,r)
    q = influence(panels;G=kelvin,Fn=0.4)\(-Uₙ.(panels;U=SA[-1,0,0]))
	d = round(1000steady_force(q,panels;G=kelvin,Fn=0.4)[1],digits=4)
	plot!(xp,ζ.(xp,yp,Ref(q),Ref(panels);G=kelvin,Fn=0.4),label="$(length(panels)) panels, 10³Cw=$d")
end
plot!(xlabel="x/L",ylabel="ζ/L")
savefig("waterline.png")
