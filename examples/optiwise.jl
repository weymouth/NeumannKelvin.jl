using NeumannKelvin
using NURBS,FileIO
patches = load("examples/optiwise_test_step.step")[[1,2,4]] # read three patches
top = maximum(p->maximum(components(p([0.,1.],[0.,1.]),3)),patches)
NURBS.translate!(patches,SA[0,0,-top]) # z=0 should be the max
length = extent(mapreduce(p->components(p([0.,1.],[0.,1.]),1),vcat,patches))

tvec = [false,true,true] # transpose a few patches for best results
h=(10,5) # set "largest" panel size
panels = mapreduce(vcat,eachindex(patches)) do i
    (hᵤ,hᵥ) = tvec[i] ? reverse(h) : h
    panelize(patches[i];hᵤ,hᵥ,transpose=tvec[i],cubature=true)
end
using GLMakie
viz(panels,panels.dA/prod(h),vectors=nothing,colormap=:ice,label="dA/hᵤhᵥ")

∫kelvin₂(x,p;kwargs...) = ∫kelvin(x,p;kwargs...)+∫kelvin(x,reflect(p,2);kwargs...)
kwargs = (ϕ=∫kelvin₂,ℓ=length/10,contour=true,filter=true)
q = influence(panels;kwargs...)\components(panels.n,1)
steady_force(q,panels;kwargs...)[1]/sum(panels.dA)
u = map(p->SA[-1,0,0]+NeumannKelvin.∇Φ(p.x,q,panels;kwargs...),panels)
plt = viz(panels,q,vectors=10u,colormap=:seismic,label=nothing)
save("optiwise.png",plt)