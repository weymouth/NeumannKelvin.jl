using NeumannKelvin
spheroid(h;L=1,r=0.25,kwargs...) = panelize(0,π,0,2π,hᵤ=h;kwargs...) do θ₁,θ₂
	SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)]
end

# Sphere area and added mass convergence
using LinearAlgebra
r = 0.5
A,V = 4π*r^2,4π/3*r^3
dat = map(0.5 .^(1:6)) do h
	panels = spheroid(h;r,N_max=13_000)
	Aerror = sum(panels.dA)/A-1
	Merror = addedmass(panels,V=0.5V)-I
	(log2h=log2(h),normAerror=norm(Aerror),normMerror=norm(Merror))
end |> Table
using CSV,Plots
CSV.write("sphere_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normMerror),label="",xlabel="log₂ h/D",ylabel="log₂ ||M error||",title="Sphere convergence")
savefig("sphere_convergence.png")

# 6:1 Spheroid added_mass convergence
r = 0.5/6.01
V = 4π/3*0.5*r^2; sol = [0.045 0 0; 0 0.918 0; 0 0 0.918]
dat = map(0.5 .^(0:0.5:3)) do h
	panels = spheroid(h*2r;r,devlimit=Inf)
	Merror = addedmass(panels;V)-sol
	(log2h=log2(h),normMerror=norm(Merror))
end |> Table
CSV.write("spheroid_ma_convergence.csv",dat)
plot(dat.log2h,log2.(dat.normMerror),label="",
	xlabel="log₂ h/2b",ylabel="log₂ ||M error||",title="6:1 spheroid M convergence")
savefig("spheroid_ma_convergence.png")

# Spheroid added_mass sweep
dat = map(0.025:0.025:0.75) do r
	panels = spheroid(min(0.5r,1/8);r)
	M = addedmass(panels)
	(r=r,M₁₁=M[1,1],M₂₂=M[2,2],M₃₃=M[3,3])
end |> Table
CSV.write("spheroid_ma_sweep.csv",dat)
plot(2dat.r,dat.M₁₁,label="M₁₁/V")
plot!(2dat.r,dat.M₂₂,label="M₂₂/V",ylims=(0,1))
plot!(2dat.r,dat.M₃₃,label="M₃₃/V")
plot!(xlabel="b/a",title="Spheroid M components")
savefig("spheroid_ma_sweep.png")

# Submerged spheroid wavemaking drag convergence
function spheroid(h;L=1,Z=-1/8,r=1/12,AR=1/2r,kwargs...)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),-r*sin(θ₂)*sin(θ₁),r*cos(θ₂)*sin(θ₁)+Z]
    panelize(S,0,π,0,π,hᵤ=h*√AR,hᵥ=h/√AR;kwargs...)
end
∫kelvin_S₂(x,p;kwargs...) = ∫kelvin(x,p;kwargs...)+∫kelvin(x,reflect(p,SA[1,-1,1]);kwargs...)
Cw(panels;kwargs...) = 2steady_force(influence(panels;kwargs...)\first.(panels.n),panels;kwargs...)[1]
kwargs = (ϕ=∫kelvin_S₂,ℓ=0.5^2)
dat = map(0.5 .^ (3:0.5:5.5)) do h
	panels = spheroid(h)
	(h=h,N=length(panels),Cw=Cw(panels;kwargs...))
end |> Table
CSV.write("submerged_spheroid_Cw_convergence.csv",dat)

kwargs = (ϕ=∫kelvin_S₂,ℓ=0.2^2)
plot(); for (n,c) in zip((20,40,60,80),colorschemes[:Blues_4])
	h = 1/n; panels = spheroid(h,N_max=1500)
    q = influence(panels;kwargs...)\components(panels.n,1);
    plot!(-2:h:1,x->ζ(x,0,q,panels;kwargs...),label="h/L=1/$n";c)
end; plot!(xlabel="x/L",ylabel="ζg/U²",ylims=(-0.15,0.15))
savefig("submerged_spheroid_WL_convergence.png")

panels = spheroid(1/60)
CwFn = map(0.15:0.05:1) do Fn
	(Fn=Fn,Cw=Cw(panels;ϕ=∫kelvin_S₂,ℓ=Fn^2))
end |> Table;
CSV.write("submerged_spheroid_Cw_Fn.csv",CwFn)
scatter(CwFn.Fn,CwFn.Cw,ylims=(0,1e-2),title="submerged spheroid wavemaking drag",
    label=nothing,xlabel="U/√gL",ylabel="Fₓ/½ρU²L²")
savefig("submerged_spheroid_Cw_Fn.png")
