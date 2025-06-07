module NeumannKelvinNURBSExt
using NeumannKelvin,NURBS

# NURBSsurface wrapper for scalar parameter arguments
scalarize(patch) = (u, v) -> patch(u, v)[1]

# Make sure exported panel functions use this wrapper
NeumannKelvin.panelize(patch::NURBSsurface,args...;kwargs...) = panelize(scalarize(patch),args...;kwargs...)
NeumannKelvin.measure_panel(patch::NURBSsurface,args...;kwargs...) = measure_panel(scalarize(patch),args...;kwargs...)
end