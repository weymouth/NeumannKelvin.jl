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

struct TriKernel <: GreenKernel end
"""
    measure(v₁, v₂, v₃) -> (x,n,dA,verts)

Measure the properties of a triangular panel defined by it's vertices in counter-clockwise order.
"""
function NeumannKelvin.measure(v₁, v₂, v₃)
    n = (v₂ - v₁) × (v₃ - v₁)
    (x=(v₁+v₂+v₃)/3, n=normalize(n), dA=norm(n)/2, verts=SA[v₁, v₂, v₃], kernel=TriKernel())
end

""" ∫G_kernel(ξ,p,::TriKernel)

Exact integrated potential over a triangular panel. See Katz and Plotkin, "Low-Speed Aerodynamics" (2001) 
"""
function NeumannKelvin.∫G_kernel(ξ, p, ::TriKernel; ignore...)
    r₁,r₂,r₃ = p.verts .- Ref(ξ); l₁,l₂,l₃ = norm.((r₁,r₂,r₃))
    @inbounds -abs(r₁'p.n)*2atan(r₁'*(r₂×r₃),l₁*l₂*l₃+r₁'r₂*l₃+r₂'r₃*l₁+r₃'r₁*l₂)
end

end