using NeumannKelvin
using NURBS,FileIO
patches = loadNURBS("examples/optiwise_test_step.step")[[1,2,4]] # read three patches
grid = range(0,1,11)
top = maximum(p->maximum(p(u,v)[3] for u in grid, v in grid),patches) # z=0 should be the max
L = extent([p(u,v)[1] for p in patches for u in grid for v in grid])

tvec = [false,true,true] # transpose a few patches for best results
h=(10,5) # set "largest" panel size
panels = mapreduce(vcat,eachindex(patches)) do i
    (hᵤ,hᵥ) = tvec[i] ? reverse(h) : h
    panelize((u,v)->patches[i](u,v)-SA[0,0,top];hᵤ,hᵥ,transpose=tvec[i],cubature=true)
end
using GLMakie
viz(panels,panels.dA/prod(h),vectors=nothing,colormap=:ice,label="dA/hᵤhᵥ")

sys = PanelSystem(panels,domain=Kelvin(L/10,sym_axes=2,contour=true,filter=true)) |> directsolve!
-steadyforce(sys)[1] # wave drag coefficient on the wetted area
plt = viz(sys,vscale=10)
save("optiwise.png",plt)
