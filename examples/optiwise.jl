using NeumannKelvin
using NURBS,FileIO
patches = load("examples/optiwise_test_step.step")[[1,2,4]] # read three patches
tvec = [false,true,true] # transpose a few patches for best results
h=(10,5) # set "largest" panel size
panels = mapreduce(vcat,eachindex(patches)) do i
    (hᵤ,hᵥ) = tvec[i] ? reverse(h) : h
    panelize(patches[i];hᵤ,hᵥ,transpose=tvec[i],cubature=true)
end
using GLMakie
viz(panels,panels.dA/prod(h),vectors=nothing,colormap=:ice,label="dA/hᵤhᵥ")