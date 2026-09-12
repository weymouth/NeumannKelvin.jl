module NeumannKelvinGeometryBasicsExt
using NeumannKelvin,GeometryBasics
using NeumannKelvin:GreenKernel,normalize
using GeometryBasics:Mesh,Point
"""
    panelize(mesh::Mesh)-> Table{panels}

Convert a Mesh of triangles into panels
"""
NeumannKelvin.panelize(mesh::Mesh) = Table([measure(mesh.position[face]) for face in mesh.faces])
NeumannKelvin.measure(verts::AbstractArray{T}) where {T<:Point{3}} = measure(SVector{3}.(verts)...)

end