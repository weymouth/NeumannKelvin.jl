using NeumannKelvin,BenchmarkTools,Test

S(θ,φ) = SA[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
panels = panelize(S, 0, π/2, 0, π, hᵤ=1/64, N_max=Inf) # quite a few panels

@btime extrema(cₚ(directsolve!(BodyPanelSystem($panels,sym_axes=(2,3)),verbose=false)))
@btime extrema(cₚ(gmressolve!(BodyPanelSystem($panels,sym_axes=(2,3)),verbose=false)))
@btime extrema(cₚ(gmressolve!(BodyPanelSystem($panels,sym_axes=(2,3),wrap=PanelTree),verbose=false)))
@btime extrema(cₚ(gmressolve!(BodyPanelSystem($panels,sym_axes=(2,3),wrap=PanelTree,θ²=1),verbose=false)))

sys = gmressolve!(BodyPanelSystem(panels,sym_axes=(2,3),wrap=PanelTree,θ²=1),verbose=false);
@btime extrema(cₚ($sys))
@btime steadyforce($sys)
