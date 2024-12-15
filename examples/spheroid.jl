using NeumannKelvin
function spheroid(h;Z=-0.5,L=1,r=0.25)
	S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
	dθ₁ = π/round(π*0.5L/h) # cosine sampling increases density at ends
	mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
		dθ₂ = π/round(π*0.25L*sin(θ₁)/h) # polar step size at this azimuth
		# param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
		param_props.(S,θ₁,0:dθ₂:2π-0.5dθ₂,dθ₁,dθ₂)
	end |> Table
end

# Sphere area and added mass convergence
using LinearAlgebra,Plots,CSV
r = 0.5
A,V = 4π*r^2,4π/3*r^3
dat = map(0.5 .^(1:6)) do h
	panels = spheroid(h;Z=0,r)
	Aerror = sum(panels.dA)/A-1
	Merror = added_mass(panels)./0.5V-I
	(log2h=log2(h),normAerror=norm(Aerror),normMerror=norm(Merror))
end |> Table
CSV.write("sphere_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normAerror),label="||A error||")
plot!(dat.log2h,log2.(dat.normMerror),label="||M error||")
plot!(xlabel="log₂ h",ylabel="log₂ error",title="Sphere convergence")
savefig("sphere_convergence.png")

# Spheroid added_mass convergence
r = 0.25
V = 4π/3*0.5*r^2; sol = [0.211115 0 0; 0 0.706404 0; 0 0 0.706404]
dat = map(0.5 .^(1:5)) do h
	panels = spheroid(h;Z=0,r)
	Merror = added_mass(panels)./V-sol
	(log2h=log2(h),normMerror=norm(Merror))
end |> Table
CSV.write("spheroid_ma_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normMerror),label="")
plot!(xlabel="log₂ h",ylabel="log₂ ||M error||",title="Spheroid M convergence")
savefig("spheroid_ma_convergence.png")

# Spheroid added_mass sweep
dat = map(0.01:0.01:0.75) do r
	panels = spheroid(1/8;Z=0,r)
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

# waterline(p) = abs(p.T₁[3]+p.T₂[3]) ≥ -2p.x[3]
# function WL_q(panels;U=SA[1,0,0],kwargs...)
#     x = panels.x |> stack
#     WL = findall(waterline,panels)
#     q = influence(panels;kwargs...)\(-Uₙ.(panels;U))
#     (x[1,WL],q[WL])
# end

# plot()
# for h in 0.5 .^(6:-1:1)
# 	scatter!(WL_q(spheroid(h;Z=0,r=0.05),U=SA[0,1,0])...,label=log2(h),alpha=0.5)
# end
# plot!(xlabel="x",ylabel="q")