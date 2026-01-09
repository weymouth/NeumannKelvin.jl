module NeumannKelvinNURBSExt
using NeumannKelvin,NURBS

# NURBSsurface wrapper for scalar parameter arguments
scalarize(patch) = (u, v) -> patch(u, v)[1]

# Make sure exported panel functions use this wrapper
NeumannKelvin.panelize(patches::AbstractArray{T},args...;kwargs...) where {T<: NURBSsurface} = mapreduce(patch->panelize(patch,args...;kwargs...),vcat,patches)
NeumannKelvin.panelize(patch::NURBSsurface,args...;kwargs...) = panelize(scalarize(patch),args...;kwargs...)
NeumannKelvin.measure(patch::NURBSsurface,args...;kwargs...) = measure(scalarize(patch),args...;kwargs...)
end