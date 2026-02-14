using NeumannKelvin,BenchmarkTools,Test

S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
for hᵤ in [1/8,1/16,1/32]
	panels = panelize(S, 0, π/2, 0, π; hᵤ, N_max=32_000) # quite a few panels
	println("Test: hᵤ=$hᵤ, N=$(length(panels))")

	println("Dense directsolve")
	@btime directsolve!(BodyPanelSystem($panels,sym_axes=(2,3)),verbose=false)
	println(" Measure")
	sys = directsolve!(BodyPanelSystem(panels,sym_axes=(2,3)),verbose=false)
	@btime extrema(cₚ($sys))
#	@btime steadyforce($sys)

	for θ² in [9,1]
		println("PanelTree θ²=$θ² gmressolve")
		@btime gmressolve!(BodyPanelSystem($panels,sym_axes=(2,3),wrap=PanelTree;θ²=$θ²),verbose=false)
		println(" Measure")
		sys = gmressolve!(BodyPanelSystem(panels,sym_axes=(2,3),wrap=PanelTree;θ²),verbose=false)
		@btime extrema(cₚ($sys))
#		@btime steadyforce($sys)
	end
end

panels = panelize(S, 0, π/2, 0, π; hᵤ=1/64, N_max=32_000) # quite a few panels
println("Stress test hᵤ=1/64, N=$(length(panels))")
@btime extrema(cₚ(gmressolve!(BodyPanelSystem($panels,sym_axes=(2,3),wrap=PanelTree;θ²=4),verbose=false)))