using NeumannKelvin
using Plots
using NURBS
using FileIO

patches = load(pkgdir(NURBS) * "/test/assets/sphere.stp")
h = 0.25
panel = measure_panel(patches[1],0.5,0.5,1.0,1.0)
panels = mapreduce(s->panelize(s;hᵤ=h),vcat,patches)
viz(panels,panels.dA/h^2,colormap=:ice,clims=(0.5,1.5))